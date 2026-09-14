(() => {
  const toast = document.querySelector(".toast");
  let timeout;

  function announce(message) {
    toast.textContent = message;
    toast.classList.add("visible");
    clearTimeout(timeout);
    timeout = window.setTimeout(() => toast.classList.remove("visible"), 2600);
  }

  for (const button of document.querySelectorAll(".copy-command")) {
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(button.dataset.copy);
        announce("Diagnostic command copied.");
      } catch {
        announce("Clipboard access was unavailable. Copy the command from the page.");
      }
    });
  }
})();
