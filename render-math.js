(function () {
  "use strict";

  // Render only marked equations and TeX delimiters; leave code and BibTeX alone.
  window.TIBRenderMath = function (root) {
    if (!window.katex || !window.renderMathInElement) return;
    root.querySelectorAll("[data-tex]").forEach(element => {
      if (element.dataset.renderedTex === element.dataset.tex) return;
      window.katex.render(element.dataset.tex, element, {
        displayMode: element.dataset.display === "true",
        throwOnError: false,
        output: "htmlAndMathml",
        // Color hooks are the only trusted extension used by the fixed equations.
        trust: context => context.command === "\\htmlClass",
        strict: code => code === "htmlExtension" ? "ignore" : "warn",
      });
      element.dataset.renderedTex = element.dataset.tex;
    });
    window.renderMathInElement(root, {
      delimiters: [
        { left: "\\(", right: "\\)", display: false },
        { left: "\\[", right: "\\]", display: true },
      ],
      throwOnError: false,
      output: "htmlAndMathml",
      ignoredClasses: ["katex", "katex-display"],
    });
  };
})();
