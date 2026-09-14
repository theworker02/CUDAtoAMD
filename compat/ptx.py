"""Fail-closed experimental straight-line PTX -> typed HIP source lowering.

Not a general PTX implementation or NVVM compiler. Only the grammar below is
accepted, and every operand is checked before generated C++ reaches HIP-Clang.
"""
import re
import tempfile
from pathlib import Path

from .compiler import compile_code_object

TYPES = {"u32": "uint32_t", "u64": "uint64_t", "f32": "float", "pred": "bool"}
NAME = r"[A-Za-z_][A-Za-z_0-9]*"


class PtxError(ValueError):
    pass


def lower_ptx(source: str) -> str:
    if len(source) > 1024 * 1024:
        raise PtxError("PTX-E001: input exceeds 1 MiB limit")
    source = re.sub(r"//[^\n]*", "", source)
    match = re.fullmatch(
        rf"\s*\.version 7\.0\s+\.target sm_50\s+\.address_size 64\s+"
        rf"\.visible\s+\.entry\s+({NAME})\s*\((.*?)\)\s*\{{(.*?)\}}\s*",
        source, re.S)
    if not match:
        raise PtxError("PTX-E002: expected one PTX 7.0/sm_50/64-bit visible entry; unsupported module syntax")
    name, params_text, body = match.groups()
    params, regs, initialized, declarations, emitted = {}, {}, set(), [], []
    for part in params_text.split(",") if params_text.strip() else []:
        param = re.fullmatch(rf"\s*\.param\s+\.(u32|u64|f32)\s+({NAME})\s*", part)
        if not param or param[2] in params:
            raise PtxError("PTX-E003: unsupported or duplicate parameter")
        params[param[2]] = param[1]
    if len(params) > 64:
        raise PtxError("PTX-E003: too many parameters")

    def register(token, kind, read=True):
        if regs.get(token) != kind or (read and token not in initialized):
            raise PtxError(f"PTX-E005: wrong type, undeclared or uninitialized register {token}")
        return "r_" + token[1:]

    def operand(token, kind):
        if kind == "u32" and re.fullmatch(r"%(tid|ctaid|ntid|nctaid)\.[xyz]", token):
            prefix, axis = token[1:].split(".")
            return {"tid": "threadIdx", "ctaid": "blockIdx", "ntid": "blockDim", "nctaid": "gridDim"}[prefix] + "." + axis
        if re.fullmatch(r"[0-9]+", token) and kind in ("u32", "u64"):
            if int(token) >= 1 << (32 if kind == "u32" else 64):
                raise PtxError("PTX-E005: immediate out of range")
            return str(int(token)) + ("u" if kind == "u32" else "ull")
        return register(token, kind)

    statements = body.split(";")
    if statements[-1].strip():
        raise PtxError("PTX-E004: missing semicolon")
    labels = {}
    for index, raw in enumerate(statements[:-1]):
        label = re.fullmatch(rf"\s*({NAME}):\s*", raw)
        if label:
            if label[1] in labels:
                raise PtxError("PTX-E004: duplicate label")
            labels[label[1]] = index
    finished = False
    for number, raw in enumerate(statements[:-1], 1):
        stmt = raw.strip()
        if finished:
            raise PtxError("PTX-E004: instructions after ret")
        label = re.fullmatch(rf"({NAME}):", stmt)
        if label:
            emitted.append(f"L_{label[1]}: ;")
            continue
        decl = re.fullmatch(r"\.reg\s+\.(u32|u64|f32|pred)\s+(%[A-Za-z_][A-Za-z_0-9]*)(?:<(\d+)>)?", stmt)
        if decl:
            if emitted:
                raise PtxError("PTX-E004: declarations must precede instructions")
            kind, base, count = decl.groups()
            if count and not 1 <= int(count) <= 256:
                raise PtxError("PTX-E004: register count outside 1..256")
            for token in ([base + str(i) for i in range(int(count))] if count else [base]):
                if token in regs or len(regs) >= 1024:
                    raise PtxError("PTX-E004: duplicate or excessive registers")
                regs[token] = kind
                declarations.append(f"{TYPES[kind]} r_{token[1:]};")
            continue
        if stmt == "ret":
            finished = True
            emitted.append("return;")
            continue
        branch = re.fullmatch(rf"@(!?%[A-Za-z_][A-Za-z_0-9]*)\s+bra\s+({NAME})", stmt)
        if branch:
            predicate, target = branch.groups()
            if labels.get(target, -1) <= number - 1:
                raise PtxError("PTX-E006: only forward predicated branches are supported")
            negate = predicate.startswith("!")
            emitted.append(f"if ({'!' if negate else ''}{register(predicate[1:] if negate else predicate, 'pred')}) goto L_{target};")
            continue
        jump = re.fullmatch(rf"bra\s+({NAME})", stmt)
        if jump:
            target = jump[1]
            if labels.get(target, -1) <= number - 1:
                raise PtxError("PTX-E006: only forward branches are supported")
            emitted.append(f"goto L_{target};")
            continue
        pieces = stmt.split(None, 1)
        if len(pieces) != 2:
            raise PtxError(f"PTX-E006: unsupported statement {number}: {stmt}")
        op, text = pieces
        args = [s.strip() for s in text.split(",")]
        kind = op.split(".")[-1]
        if op in ("ld.param.u32", "ld.param.u64", "ld.param.f32") and len(args) == 2:
            param = re.fullmatch(rf"\[({NAME})\]", args[1])
            if not param or params.get(param[1]) != kind:
                raise PtxError("PTX-E005: parameter type/name mismatch")
            expr = "p_" + param[1]
        elif op in ("mov.u32", "mov.u64", "mov.f32") and len(args) == 2:
            expr = operand(args[1], kind)
        elif op in ("add.u32", "add.u64", "sub.u32", "sub.u64", "mul.lo.u32", "mul.lo.u64", "and.b32", "or.b32", "xor.b32", "add.rn.f32") and len(args) == 3:
            if op.endswith("b32"):
                kind = "u32"
            left, right = operand(args[1], kind), operand(args[2], kind)
            if kind == "f32":
                expr = f"__fadd_rn({left}, {right})"
            else:
                operator = '*' if op.startswith("mul") else '-' if op.startswith("sub") else '&' if op.startswith("and") else '|' if op.startswith("or") else '^' if op.startswith("xor") else '+'
                expr = f"({left} {operator} {right})"
        elif op == "mul.rn.f32" and len(args) == 3:
            kind = "f32"
            expr = f"__fmul_rn({operand(args[1], kind)}, {operand(args[2], kind)})"
        elif op == "fma.rn.f32" and len(args) == 4:
            kind = "f32"
            expr = f"__fmaf_rn({operand(args[1], kind)}, {operand(args[2], kind)}, {operand(args[3], kind)})"
        elif op == "mul.wide.u32" and len(args) == 3:
            kind = "u64"
            expr = f"(static_cast<uint64_t>({operand(args[1], 'u32')}) * static_cast<uint64_t>({operand(args[2], 'u32')}))"
        elif op == "cvt.u64.u32" and len(args) == 2:
            kind = "u64"
            expr = f"static_cast<uint64_t>({operand(args[1], 'u32')})"
        elif op == "cvt.u32.u64" and len(args) == 2:
            kind = "u32"
            expr = f"static_cast<uint32_t>({operand(args[1], 'u64')})"
        elif op == "cvt.rn.f32.u32" and len(args) == 2:
            kind = "f32"
            expr = f"static_cast<float>({operand(args[1], 'u32')})"
        elif op in ("setp.lt.u32", "setp.le.u32", "setp.eq.u32", "setp.ne.u32", "setp.ge.u32", "setp.gt.u32") and len(args) == 3:
            kind = "pred"
            comparison = {"setp.lt.u32": "<", "setp.le.u32": "<=", "setp.eq.u32": "==", "setp.ne.u32": "!=", "setp.ge.u32": ">=", "setp.gt.u32": ">"}[op]
            expr = f"({operand(args[1], 'u32')} {comparison} {operand(args[2], 'u32')})"
        elif op in ("ld.global.f32", "st.global.f32", "ld.global.u32", "st.global.u32") and len(args) == 2:
            kind = op.rsplit(".", 1)[1]
            address_index = 0 if op.startswith("st") else 1
            address = re.fullmatch(r"\[(%[A-Za-z_][A-Za-z_0-9]*)\]", args[address_index])
            if not address:
                raise PtxError("PTX-E005: unsupported memory addressing")
            pointer = register(address[1], "u64")
            expr = f"*reinterpret_cast<{TYPES[kind]}*>({pointer})"
            if address_index == 0:
                emitted.append(f"{expr} = {operand(args[1], kind)};")
                continue
        else:
            raise PtxError(f"PTX-E006: unsupported instruction {number}: {op}")
        dest = register(args[0], kind, read=False)
        emitted.append(f"{dest} = {expr};")
        initialized.add(args[0])
    if not finished:
        raise PtxError("PTX-E004: entry must end with ret")
    signature = ", ".join(f"{TYPES[t]} p_{p}" for p, t in params.items())
    return '\n'.join(['#include <stdint.h>', '#include <hip/hip_runtime.h>',
                      f'extern "C" __global__ void {name}({signature}) {{',
                      *declarations, *emitted, '}'])


def compile_ptx(source, output, arch, *, compiler):
    """Explicit AOT translation; no runtime interception or persistent cache."""
    source = Path(source).resolve(strict=True)
    with source.open("r", encoding="utf-8") as stream:
        lowered = lower_ptx(stream.read(1024 * 1024 + 1))
    with tempfile.TemporaryDirectory(prefix="compat-ptx-") as directory:
        generated = Path(directory) / "lowered.hip"
        generated.write_text(lowered, encoding="utf-8")
        return compile_code_object(generated, output, arch, compiler=compiler)
