"""Run after building vector_add.hsaco; accepts an explicit trusted module path."""
import argparse
from array import array
from compat.native import Device, Event, KernelArg, MemoryPool, Module, Stream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("module")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    with Device(args.device) as device, Stream(device) as stream, MemoryPool(device) as pool:
        print(device.properties)
        n = 257
        a = array("f", (i * .25 for i in range(n)))
        b = array("f", (i % 7 - 3 for i in range(n)))
        out = array("f", [0] * n)
        with Module(device, args.module) as module, Event(device) as start, Event(device) as stop:
            with pool.buffer(n * 4, stream) as da, pool.buffer(n * 4, stream) as db, pool.buffer(n * 4, stream) as dc:
                da.copy_from_host_async(a)
                db.copy_from_host_async(b)
                start.record(stream)
                module.launch(module.get_function("vector_add"), (3,1,1), (128,1,1),
                              stream, [da, db, dc, KernelArg("int32_t", n)])
                stop.record(stream)
                dc.copy_to_host_async(out)
                stream.synchronize()
                if list(out) != [x + y for x, y in zip(a, b)]:
                    raise AssertionError("GPU results differ from CPU reference")
                print(f"Verified {n} outputs; kernel interval {start.elapsed_ms(stop):.3f} ms")


if __name__ == "__main__":
    main()
