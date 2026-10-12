(function () {
  "use strict";
  const byId = id => document.getElementById(id);
  const math = window.TIBMath;
  window.TIBRenderMath(document.body);
  document.querySelectorAll(".hero-actions a").forEach(link => {
    if (!link.getAttribute("href")) link.setAttribute("aria-disabled", "true");
    link.addEventListener("click", event => {
      if (!link.getAttribute("href")) event.preventDefault();
    });
  });
  const captions = {
    aligned: "A man with four running dogs in nature.",
    misaligned: "A bicycler is riding a bike at a racing event.",
  };
  const methods = {
    tib: { name: "TIB", x: 338 },
    original: { name: "Input image", x: 46 },
    nib: { name: "NIB", x: 629 },
    m2ib: { name: "M2IB", x: 922 },
    gradcam: { name: "Grad-CAM", x: 1214 },
    fastig: { name: "Fast-IG", x: 1506 },
    chefer: { name: "Chefer", x: 1798 },
  };
  let pair = "aligned";
  let baseline = "m2ib";
  function setCrop(element, method, alignment) {
    const image = element.querySelector("img");
    const crop = 250;
    const x = methods[method].x;
    const y = alignment === "aligned" ? 74 : 407;
    image.style.width = `${2059 / crop * 100}%`;
    image.style.left = `${-x / crop * 100}%`;
    image.style.top = `${-y / crop * 100}%`;
    element.dataset.crop = method;
    element.dataset.pair = alignment;
  }
  document.querySelectorAll(".paper-crop").forEach(element => setCrop(element, element.dataset.crop, element.dataset.pair));
  function renderComparison() {
    const aligned = pair === "aligned";
    const stateLabel = aligned ? "Aligned pair" : "Misaligned pair";
    const methodName = methods[baseline].name;
    for (const id of ["pair-status"]) {
      byId(id).textContent = stateLabel;
      byId(id).className = `alignment-label ${pair}`;
    }
    byId("comparison-caption").textContent = `“${captions[pair]}”`;
    document.querySelector(".caption-hint").textContent = aligned ? "The caption describes the scene." : "The caption describes a different scene.";
    document.querySelectorAll("[data-pair-button]").forEach(button => {
      const active = button.dataset.pairButton === pair;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    setCrop(byId("tib-map"), "tib", pair);
    setCrop(byId("baseline-map"), baseline, pair);
    byId("tib-map").querySelector("img").alt = `TIB image attribution for the ${pair} caption`;
    byId("baseline-map").querySelector("img").alt = `${methodName} image attribution for the ${pair} caption`;
    byId("baseline-title").textContent = methodName;
    byId("tib-observation").textContent = aligned
      ? "The paper example highlights the person and dogs under aligned supervision."
      : "In this paper example, the mismatched caption produces a much weaker visible TIB response.";
    const observations = {
      m2ib: aligned ? "A fixed-variance information bottleneck, shown on the same input." : "The fixed-variance bottleneck still shows prominent regions for this unrelated caption.",
      nib: aligned ? "NIB's deterministic narrowing attribution on the aligned pair." : "NIB still shows prominent responses in this mismatched paper example.",
      gradcam: aligned ? "The Grad-CAM example concentrates on a small region of the scene." : "Localized responses remain visible with the unrelated caption.",
      fastig: aligned ? "The paper's Fast-IG visualization on the same aligned pair." : "A broad response remains visible in this mismatched example.",
      chefer: aligned ? "The paper's attention-based Chefer visualization on the aligned pair." : "Attention-based regions remain visible with the unrelated caption.",
    };
    byId("baseline-observation").textContent = observations[baseline];
    byId("comparison-takeaway").textContent = aligned
      ? "A meaningful caption can support a meaningful explanation. The challenge is what happens when that correspondence disappears."
      : "A visible heatmap does not establish that a caption is trustworthy. TIB models sample-wise uncertainty; this example illustrates its intended behavior.";
  }
  document.querySelectorAll("[data-pair-button]").forEach(button => button.addEventListener("click", () => { pair = button.dataset.pairButton; renderComparison(); }));
  byId("baseline").addEventListener("change", event => { baseline = event.target.value; renderComparison(); });
  renderComparison();

  const controlIds = ["beta", "alpha", "log-variance", "mask-logit", "residual", "sample-count", "sample-seed"];
  const defaults = { beta: .1, alpha: .5, "log-variance": 0, "mask-logit": 0, residual: .25, "sample-count": 5, "sample-seed": 42 };
  let latestResult;
  let latestSamples;
  const number = value => Number(value).toFixed(3);
  function drawObjective(result, values) {
    const components = [
      { label: "Fitting", formula: "\\(r(\\sigma^2)^{\\alpha-1}\\)", value: result.fittingContribution, color: "#2866a6" },
      { label: "Variance term", formula: "\\(\\frac{1}{2}(\\sigma^2)^\\alpha\\log\\sigma^2\\)", value: result.normalization, color: "#8493a5" },
      { label: "Compression", formula: "\\(\\beta\\cdot\\mathrm{KL}\\)", value: result.compression, color: "#b56b54" },
    ];
    const minimum = Math.min(-0.5, ...components.map(component => component.value));
    const maximum = Math.max(1, ...components.map(component => component.value));
    const x = value => 145 + (value - minimum) / (maximum - minimum) * 300;
    const zero = x(0);
    let loss = `<line x1="${zero}" y1="12" x2="${zero}" y2="121" stroke="#c9d1db"/>`;
    components.forEach((component, index) => {
      const y = 27 + index * 38;
      const end = x(component.value);
      loss += `<g class="loss-component" data-component="${component.label}" data-value="${component.value}"><text x="0" y="${y}" fill="#30343b" font-size="12">${component.label}</text><foreignObject x="0" y="${y + 3}" width="140" height="23"><div xmlns="http://www.w3.org/1999/xhtml" class="chart-math">${component.formula}</div></foreignObject><rect class="loss-bar" x="${Math.min(zero, end)}" y="${y - 10}" width="${Math.abs(end - zero)}" height="15" rx="2" fill="${component.color}"/><text x="515" y="${y + 2}" fill="${component.color}" font-size="12" text-anchor="end">${number(component.value)}</text></g>`;
    });
    loss += `<line x1="145" y1="126" x2="445" y2="126" stroke="#e3e7ec"/>`;
    const ticks = new Set([minimum, 0, maximum]);
    ticks.forEach(tick => { loss += `<text x="${x(tick)}" y="141" fill="#68717c" font-size="10" text-anchor="middle">${tick.toFixed(2)}</text>`; });
    byId("loss-chart").innerHTML = loss;
    window.TIBRenderMath(byId("loss-chart"));
    byId("loss-chart").setAttribute("aria-label", `Loss contributions: fitting ${number(result.fittingContribution)}, variance term ${number(result.normalization)}, compression ${number(result.compression)}. Signed axis from ${minimum.toFixed(2)} to ${maximum.toFixed(2)}.`);

    const px = logVariance => 52 + (logVariance + 3) / 6 * 440;
    const py = logCoefficient => 115 - (logCoefficient + 3) / 6 * 94;
    let sensitivity = "";
    for (const tick of [-3, 0, 3]) {
      sensitivity += `<line x1="${px(tick)}" y1="21" x2="${px(tick)}" y2="115" stroke="#edf0f4"/><line x1="52" y1="${py(tick)}" x2="492" y2="${py(tick)}" stroke="#edf0f4"/><text x="${px(tick)}" y="133" fill="#68717c" font-size="10" text-anchor="middle">${Math.exp(tick).toFixed(tick < 0 ? 2 : 1)}</text><text x="43" y="${py(tick) + 3}" fill="#68717c" font-size="10" text-anchor="end">${Math.exp(tick).toFixed(tick < 0 ? 2 : 1)}</text>`;
    }
    const curve = alpha => Array.from({ length: 61 }, (_, index) => {
      const logVariance = -3 + index / 10;
      return `${px(logVariance)},${py((alpha - 1) * logVariance)}`;
    }).join(" ");
    sensitivity += `<text x="52" y="12" fill="#68717c" font-size="10">Direct fitting coefficient</text><text x="492" y="12" fill="#68717c" font-size="9" text-anchor="end">logarithmic axes</text><polyline points="${curve(.5)}" fill="none" stroke="#b8c2ce" stroke-dasharray="4 4" stroke-width="1.5"/><polyline id="sensitivity-curve" points="${curve(values.alpha)}" fill="none" stroke="#2866a6" stroke-width="2.5"/><circle cx="${px(values["log-variance"])}" cy="${py((values.alpha - 1) * values["log-variance"])}" r="4" fill="#2866a6" stroke="#fff" stroke-width="1.5"/><foreignObject x="52" y="140" width="440" height="24"><div xmlns="http://www.w3.org/1999/xhtml" class="chart-math chart-axis-label">Predicted variance \\(\\sigma^2\\) · dashed: \\(\\alpha=0.5\\)</div></foreignObject>`;
    byId("sensitivity-chart").innerHTML = sensitivity;
    window.TIBRenderMath(byId("sensitivity-chart"));
    byId("sensitivity-chart").setAttribute("aria-label", `Direct fitting coefficient versus variance, on logarithmic axes. Alpha ${values.alpha.toFixed(2)} gives log-log slope ${(values.alpha - 1).toFixed(2)}. Current variance ${number(result.variance)}, coefficient ${number(result.fittingCoefficient)}. All curves pass through variance 1, coefficient 1.`);
  }
  function drawDistribution(result, points) {
    const canvas = byId("distribution-chart");
    const width = canvas.getBoundingClientRect().width;
    if (width <= 0) return;
    const height = 230;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const context = canvas.getContext("2d");
    if (!context) return;
    context.scale(ratio, ratio);
    const left = 36, right = width - 12, top = 38, bottom = 188;
    const minX = Math.min(-2, result.mean - result.std * 3.7);
    const maxX = Math.max(2.5, result.mean + result.std * 3.7);
    const px = value => left + (value - minX) / (maxX - minX) * (right - left);
    const py = value => bottom - value * (bottom - top);
    context.font = "9px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
    context.fillStyle = "#686b73";
    context.textAlign = "left";
    context.fillText("relative density (peak = 1)", left, 17);
    context.textAlign = "right";
    context.fillText("z", right, height - 4);
    for (let tick = 0; tick <= 4; tick++) {
      const value = minX + tick / 4 * (maxX - minX);
      const x = px(value);
      context.strokeStyle = "#edece5";
      context.beginPath(); context.moveTo(x, top); context.lineTo(x, bottom); context.stroke();
      context.textAlign = "center"; context.fillStyle = "#777982";
      context.fillText(value.toFixed(1), x, bottom + 29);
    }
    context.strokeStyle = "#dcded6";
    context.beginPath(); context.moveTo(left, bottom); context.lineTo(right, bottom); context.stroke();
    context.beginPath();
    // Include the exact center so very narrow distributions do not vanish between plot samples.
    const xValues = Array.from({ length: 301 }, (_, index) => minX + index / 300 * (maxX - minX));
    xValues.push(result.mean); xValues.sort((a, b) => a - b);
    context.moveTo(left, bottom);
    xValues.forEach(value => {
      const normalizedDensity = Math.exp(-0.5 * Math.pow((value - result.mean) / result.std, 2));
      context.lineTo(px(value), py(normalizedDensity));
    });
    context.lineTo(right, bottom); context.closePath();
    const gradient = context.createLinearGradient(0, top, 0, bottom);
    gradient.addColorStop(0, "#2866a630"); gradient.addColorStop(1, "#2866a604");
    context.fillStyle = gradient; context.fill();
    context.beginPath();
    xValues.forEach((value, index) => {
      const y = py(Math.exp(-0.5 * Math.pow((value - result.mean) / result.std, 2)));
      if (index === 0) context.moveTo(px(value), y); else context.lineTo(px(value), y);
    });
    context.lineWidth = 2; context.strokeStyle = "#2866a6"; context.stroke(); context.lineWidth = 1;
    context.setLineDash([4, 4]); context.strokeStyle = "#b56b54";
    context.beginPath(); context.moveTo(px(1), top - 7); context.lineTo(px(1), bottom); context.stroke();
    context.setLineDash([]); context.fillStyle = "#b56b54"; context.textAlign = "center";
    context.fillText("f = 1", px(1), top - 15);
    points.forEach((value, index) => {
      const x = px(value);
      if (x < left || x > right) return;
      context.beginPath(); context.arc(x, bottom + 8 + index % 3 * 3, 2.4, 0, Math.PI * 2);
      context.fillStyle = "#2866a699"; context.fill();
    });
    canvas.setAttribute("aria-label", `Bottleneck Gaussian: mean ${number(result.mean)}, standard deviation ${number(result.std)}; ${points.length} Monte Carlo samples. The density curve is normalized to its peak.`);
  }
  function renderLab() {
    const values = Object.fromEntries(controlIds.map(id => [id, Number(byId(id).value)]));
    const seedField = byId("sample-seed");
    let seed = values["sample-seed"];
    if (!Number.isInteger(seed) || seed < 0 || seed > 2147483647 || seedField.value.trim() === "") {
      seedField.setAttribute("aria-invalid", "true");
      seed = 42;
    } else seedField.removeAttribute("aria-invalid");
    const result = math.mechanism({ beta: values.beta, alpha: values.alpha, logVariance: values["log-variance"], maskLogit: values["mask-logit"], residual: values.residual });
    const points = math.samples(result.mean, result.std, values["sample-count"], seed);
    latestResult = result; latestSamples = points;
    byId("beta-value").textContent = values.beta.toFixed(2);
    byId("alpha-value").textContent = values.alpha.toFixed(2);
    byId("log-variance-value").textContent = result.variance.toFixed(2);
    byId("mask-logit-value").textContent = result.h.toFixed(2);
    byId("residual-value").textContent = values.residual.toFixed(2);
    byId("sample-count-value").textContent = String(values["sample-count"]);
    byId("retained-mean").textContent = number(result.mean);
    byId("noise-std").textContent = number(result.std);
    byId("sample-mean").textContent = number(points.reduce((sum, value) => sum + value, 0) / points.length);
    byId("fitting-coefficient").textContent = number(result.fittingCoefficient);
    byId("kl-value").textContent = number(result.kl);
    byId("likelihood-value").textContent = number(result.likelihood);
    byId("compression-value").textContent = number(result.compression);
    byId("total-loss").textContent = number(result.total);
    let note;
    if (Math.abs(values["log-variance"]) < 0.01) note = "At \\(\\sigma^2=1\\), the direct fitting coefficient is 1 for every \\(\\alpha\\).";
    else if (values.alpha === 1) note = "At \\(\\alpha=1\\), the direct fitting coefficient is constant. Variance still changes the noise, likelihood, and KL.";
    else if (result.variance > 1) note = "With \\(\\alpha<1\\), higher variance reduces the direct fitting coefficient. This does not determine the complete mask gradient.";
    else note = "With \\(\\alpha<1\\), lower variance increases the direct fitting coefficient. The mask and KL also contribute to optimization.";
    byId("mechanism-note").textContent = note;
    window.TIBRenderMath(byId("mechanism-note"));
    drawObjective(result, values);
    drawDistribution(result, points);
  }
  controlIds.forEach(id => byId(id).addEventListener("input", renderLab));
  byId("sample-seed").addEventListener("blur", () => {
    if (byId("sample-seed").getAttribute("aria-invalid") === "true") { byId("sample-seed").value = 42; renderLab(); }
  });
  byId("resample").addEventListener("click", () => { byId("sample-seed").value = (Number(byId("sample-seed").value) + 1) % 2147483648; renderLab(); });
  byId("reset-lab").addEventListener("click", () => { Object.entries(defaults).forEach(([id, value]) => { byId(id).value = value; }); renderLab(); });
  byId("try-variance").addEventListener("click", () => { byId("log-variance").value = 1.38; renderLab(); });
  let resizeFrame;
  window.addEventListener("resize", () => { cancelAnimationFrame(resizeFrame); resizeFrame = requestAnimationFrame(() => drawDistribution(latestResult, latestSamples)); });
  renderLab();

  // All fields correspond to the source AttributionConfig; learned quantities are explained separately.
  const parameters = [
    { key: "beta", group: "objective", default: "0.1", title: "The cost of retaining information", role: "Multiplies the KL compression term. The implemented KL is to a standard normal prior and is averaged over valid feature dimensions during optimization.", effect: "Increasing \\(\\beta\\) makes the same retained-information cost more expensive. It does not directly prescribe a heatmap intensity or guarantee better attribution.", target: "beta" },
    { key: "alpha", group: "objective", default: "0.5", title: "How variance scales the fitting gradient", role: "The detached weight is \\((\\sigma^2)^\\alpha\\). Combined with the inverse variance in the likelihood, the direct fitting coefficient becomes \\((\\sigma^2)^{\\alpha-1}\\). Valid values are between 0 and 1.", effect: "At \\(\\alpha=0\\), direct fitting scales as \\(\\frac{1}{\\sigma^2}\\). At \\(\\alpha=1\\), this coefficient is constant. The weight is detached, but the variance is still optimized through other terms.", target: "alpha" },
    { key: "fitting", group: "objective", default: '"cosine"', title: "The cross-modal agreement term", role: "Chooses either cosine similarity minus 1 or negative half the squared distance between normalized prediction and target embeddings. These two forms are mathematically equivalent on unit embeddings.", effect: "This is not raw positive cosine in the TIB likelihood: the offset makes the fitting term consistent with the normalized Gaussian interpretation. Both supported modes express the same fit." },
    { key: "target_layer", group: "bottleneck", default: "9", title: "Where the bottleneck enters the encoder", role: "Selects a zero-based Transformer block index. The vision and text branches each insert their own bottleneck at the chosen layer boundary.", effect: "Changing the layer changes the feature representation being explained. Earlier and later boundaries expose different representations; a later layer is not automatically better.", widget: "layers" },
    { key: "layer_position", group: "bottleneck", default: '"input"', title: "Before or after the selected block", role: "The bottleneck temporarily replaces the input or output of the selected Transformer block. Native attention masks, pooling, and projection operations remain in the forward path.", effect: "With target_layer = 9, input means just before block 9; output means just after it. Changing this setting changes the actual representation being perturbed." },
    { key: "mask_init", group: "bottleneck", default: "0.0", title: "The starting point of the retention mask", role: "Initializes every trainable mask logit. The retained fraction is \\(h=\\operatorname{sigmoid}(\\text{logit})\\), so the default 0 gives \\(h=0.5\\).", effect: "A larger initial logit starts closer to keeping the feature; a smaller one starts closer to replacing it with noise. The real mask is then optimized feature by feature.", target: "mask-logit" },
    { key: "variance_hidden_dim", group: "bottleneck", default: "128", title: "Capacity of the variance predictor", role: "Sets the hidden width of the per-pair MLP: LayerNorm(D) → Linear(D, H) → GELU → Linear(H, 1). Valid tokens are pooled before predicting one scalar log variance.", effect: "A larger width adds parameters and compute. It does not change the output from a sample-wise scalar into a feature-wise variance, nor does it ensure improved uncertainty estimates." },
    { key: "min_log_variance", group: "bottleneck", default: "−8.0", title: "The lower numerical bound on variance", role: "Defines the lower log-variance bound in the predictor's smooth tanh mapping. Bounds must satisfy \\(-30\\leq\\mathrm{min}<\\mathrm{max}\\leq30\\).", effect: "A smaller lower bound permits a smaller \\(\\sigma^2\\) and potentially stronger inverse-variance fitting pressure. The mapping is smooth; the bound is not a hard clamp on the prediction." },
    { key: "max_log_variance", group: "bottleneck", default: "8.0", title: "The upper numerical bound on variance", role: "Defines the upper log-variance bound in the smooth tanh mapping. Variance is exp(log variance) and is broadcast over the feature dimensions of a single input.", effect: "A larger upper bound allows more noise and a wider range of uncertainty scaling. These bounds are numerical choices, not an automatic calibration guarantee." },
    { key: "steps", group: "optimization", default: "10", title: "How long the per-pair bottleneck is fitted", role: "Sets the number of Adam updates for each modality's mask and variance predictor. The pretrained encoder remains frozen.", effect: "More steps allow more optimization and take longer. They do not retrain the encoder and do not guarantee a more faithful explanation." },
    { key: "lr", group: "optimization", default: "0.01", title: "How far each mask update can move", role: "Sets the initial Adam learning rate for the mask. If variance_lr is None, the variance predictor uses this learning rate too.", effect: "A higher rate makes larger updates possible and can change convergence or stability. A selected mask schedule can decay this rate over the per-pair steps.", widget: "schedule" },
    { key: "variance_lr", group: "optimization", default: "None", title: "An independent pace for uncertainty learning", role: "An optional learning rate for the variance predictor's Adam parameter group. None means use lr.", effect: "Setting a separate value changes how quickly the scalar-variance branch can move relative to the mask. Its rate stays constant even when the mask rate is scheduled." },
    { key: "mask_lr_schedule", group: "optimization", default: '"constant"', title: "How the mask learning rate changes over time", role: "Chooses constant, linear, or cosine decay. The first step uses lr; decayed schedules end at lr × mask_lr_final_ratio. A single-step run uses lr.", effect: "A schedule can make later mask updates smaller while the variance predictor keeps its own constant rate. The plot below follows the implementation exactly.", widget: "schedule" },
    { key: "mask_lr_final_ratio", group: "optimization", default: "0.1", title: "The final scale of a decayed mask update", role: "Defines the fraction of the initial mask learning rate reached on the final step of linear or cosine decay. Allowed values are between 0 and 1.", effect: "A lower ratio gives smaller final updates. It has no effect under the constant schedule or when steps = 1.", widget: "schedule" },
    { key: "mask_adam_beta1", group: "optimization", default: "0.9", title: "Smoothing of the mask's gradient history", role: "Controls the exponential moving average of first-order gradients in the mask's Adam parameter group. Values must be in \\([0,1)\\).",
      effect: "A larger value gives more memory to previous gradients; a smaller value responds faster to new gradients. It does not change the variance predictor's Adam defaults." },
    { key: "mask_adam_beta2", group: "optimization", default: "0.999", title: "Smoothing of the mask's squared-gradient history", role: "Controls Adam's second-moment estimate for the mask parameter group. This is an optimizer setting, distinct from the compression coefficient \\(\\beta\\).", effect: "A larger value keeps a longer history of squared gradients. Changing it affects adaptive update scaling; it does not directly alter the bottleneck distribution." },
    { key: "mask_adam_eps", group: "optimization", default: "1e−8", title: "Numerical stability in the mask optimizer", role: "Adds a small positive constant to Adam's denominator for the mask parameter group. It must be positive and finite.", effect: "A larger epsilon can reduce update magnitudes when estimated gradients are small. This setting affects the optimizer, not the variance bound or the KL formula." },
    { key: "num_samples", group: "sampling", default: "5", title: "How many noisy representations estimate the fit", role: "Repeats the input K times and draws K bottleneck samples per update to estimate the fitting term by Monte Carlo averaging.", effect: "More samples generally reduce sampling noise at additional compute cost. They do not change \\(q(z\\mid x)\\), the number of real input pairs, or the learned target dimension.", target: "sample-count" },
    { key: "seed", group: "sampling", default: "42", title: "Reproducible initialization and noise", role: "Seeds bottleneck initialization and noise. The original RNG state is restored afterward. With independent branch configs, each modality resets to its own configured seed.", effect: "Changing the seed changes the random draws and can change the fitted attribution. Keeping it fixed makes repeated calls on the same setup reproducible.", target: "sample-seed" },
  ];
  let filter = "all";
  let selectedParameter = "beta";
  let selectedLayer = 9;
  let selectedSchedule = "constant";
  function scheduleMarkup() {
    const width = 400, height = 104, left = 28, top = 13, right = 390, bottom = 76;
    const points = Array.from({ length: 10 }, (_, step) => {
      const rate = math.learningRate(.01, .1, step, 10, selectedSchedule);
      return `${left + step / 9 * (right - left)},${bottom - rate / .01 * (bottom - top)}`;
    }).join(" ");
    return `<div class="schedule-preview"><div class="schedule-controls" role="group" aria-label="Preview a mask learning rate schedule">${["constant", "linear", "cosine"].map(schedule => `<button data-schedule="${schedule}" class="${schedule === selectedSchedule ? "is-active" : ""}" aria-pressed="${schedule === selectedSchedule}">${schedule}</button>`).join("")}</div><svg class="schedule-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${selectedSchedule} mask learning rate over ten updates, from 0.01 with final ratio 0.1"><line x1="${left}" y1="${bottom}" x2="${right}" y2="${bottom}" stroke="#dddbe7"/><polyline points="${points}" fill="none" stroke="#2866a6" stroke-width="2.5"/><g fill="#686b73" font-family="sans-serif" font-size="9"><text x="${left}" y="97">Step 1</text><text x="${right}" y="97" text-anchor="end">Step 10</text><text x="0" y="16">.01</text><text x="0" y="${bottom}">0</text></g></svg><p class="schedule-caption">Illustration: lr = 0.01, steps = 10, final ratio = 0.1. Variance learning rate is constant.</p></div>`;
  }
  function renderDetail() {
    const parameter = parameters.find(item => item.key === selectedParameter);
    if (!parameter) return;
    const layers = parameter.widget === "layers" ? `<div class="encoder-strip" role="group" aria-label="Select an illustrative CLIP layer">${Array.from({ length: 12 }, (_, index) => `<button class="${selectedLayer === index ? "is-active" : ""}" data-layer="${index}" aria-pressed="${selectedLayer === index}" aria-label="Zero-based encoder block ${index}">${index}</button>`).join("")}</div><p class="encoder-caption">Selected: block ${selectedLayer}, input boundary. A location illustration, not a new attribution run.</p>` : "";
    byId("parameter-detail").innerHTML = `<p class="eyebrow">${parameter.group} / Configuration</p><h3>${parameter.title}</h3><span class="default-pill"><code>${parameter.key}</code> · default ${parameter.default}</span><p>${parameter.role}</p><div class="detail-callout"><strong>What changes?</strong>${parameter.effect}</div>${layers}${parameter.widget === "schedule" ? scheduleMarkup() : ""}<div class="detail-footer"><span>Source: AttributionConfig</span>${parameter.target ? `<button class="text-button" data-focus="${parameter.target}">Try the control <span aria-hidden="true">↗</span></button>` : ""}</div>`;
    window.TIBRenderMath(byId("parameter-detail"));
    byId("parameter-detail").querySelectorAll("[data-layer]").forEach(button => button.addEventListener("click", () => { selectedLayer = Number(button.dataset.layer); renderDetail(); byId("parameter-detail").querySelector(`[data-layer="${selectedLayer}"]`).focus(); }));
    byId("parameter-detail").querySelectorAll("[data-schedule]").forEach(button => button.addEventListener("click", () => { selectedSchedule = button.dataset.schedule; renderDetail(); byId("parameter-detail").querySelector(`[data-schedule="${selectedSchedule}"]`).focus(); }));
    byId("parameter-detail").querySelectorAll("[data-focus]").forEach(button => button.addEventListener("click", () => {
      const target = byId(button.dataset.focus);
      target.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" });
      target.focus({ preventScroll: true });
    }));
  }
  function renderParameters() {
    const query = byId("parameter-search").value.toLowerCase().trim();
    const visible = parameters.filter(parameter => (filter === "all" || filter === parameter.group) && `${parameter.key} ${parameter.title} ${parameter.role}`.toLowerCase().includes(query));
    if (!visible.length) {
      byId("parameter-list").innerHTML = '<p class="empty-parameters">No matching parameters. Try a different name or category.</p>';
      byId("parameter-detail").innerHTML = '<p class="eyebrow">Field guide</p><h3>No matching parameters.</h3><p>Clear the search or choose All to explore the available configuration fields.</p>';
      return;
    }
    if (!visible.some(parameter => parameter.key === selectedParameter)) selectedParameter = visible[0].key;
    byId("parameter-list").innerHTML = visible.map(parameter => `<button class="parameter-item ${parameter.key === selectedParameter ? "is-active" : ""}" data-parameter="${parameter.key}" aria-pressed="${parameter.key === selectedParameter}"><span><code>${parameter.key}</code><small>${parameter.group}</small></span><span class="param-default">${parameter.default}</span></button>`).join("");
    byId("parameter-list").querySelectorAll("[data-parameter]").forEach(button => button.addEventListener("click", () => {
      selectedParameter = button.dataset.parameter;
      byId("parameter-list").querySelectorAll("[data-parameter]").forEach(item => { const active = item.dataset.parameter === selectedParameter; item.classList.toggle("is-active", active); item.setAttribute("aria-pressed", String(active)); });
      renderDetail();
    }));
    renderDetail();
  }
  // Use the actual list length to keep the field-guide introduction accurate.
  document.querySelector(".parameter-section .section-heading > p:last-child").innerHTML = `Explore all ${parameters.length} configuration fields. Defaults come from <code>AttributionConfig</code>; the downloadable code also includes the selected CLIP ViT-B/32 branch profile.`;
  document.querySelectorAll("[data-filter]").forEach(button => button.addEventListener("click", () => {
    filter = button.dataset.filter;
    document.querySelectorAll("[data-filter]").forEach(item => { const active = item.dataset.filter === filter; item.classList.toggle("is-active", active); item.setAttribute("aria-pressed", String(active)); });
    renderParameters();
  }));
  byId("parameter-search").addEventListener("input", renderParameters);
  renderParameters();

  byId("menu-toggle").addEventListener("click", () => {
    const open = byId("menu-toggle").getAttribute("aria-expanded") !== "true";
    byId("menu-toggle").setAttribute("aria-expanded", String(open));
    byId("navigation").classList.toggle("is-open", open);
  });
  byId("navigation").querySelectorAll("a").forEach(link => link.addEventListener("click", () => { byId("navigation").classList.remove("is-open"); byId("menu-toggle").setAttribute("aria-expanded", "false"); }));
  document.addEventListener("keydown", event => { if (event.key === "Escape") { byId("navigation").classList.remove("is-open"); byId("menu-toggle").setAttribute("aria-expanded", "false"); } });
  let copyTimeout;
  byId("copy-citation").addEventListener("click", async () => {
    const citation = byId("citation-text").textContent;
    let copied = false;
    try {
      if (navigator.clipboard && window.isSecureContext) { await navigator.clipboard.writeText(citation); copied = true; }
    } catch { /* The local-file fallback below also works without clipboard permission. */ }
    if (!copied) {
      const text = document.createElement("textarea");
      text.value = citation; text.style.position = "fixed"; text.style.opacity = "0";
      document.body.appendChild(text); text.select();
      try { copied = document.execCommand("copy"); } catch { copied = false; }
      text.remove(); byId("copy-citation").focus();
    }
    byId("copy-citation").textContent = copied ? "Copied!" : "Select BibTeX";
    byId("copy-status").textContent = copied ? "BibTeX copied to clipboard." : "Select the citation text, or use the Download .bib link.";
    if (!copied) { const range = document.createRange(); range.selectNodeContents(byId("citation-text")); const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range); }
    clearTimeout(copyTimeout); copyTimeout = setTimeout(() => { byId("copy-citation").textContent = "Copy BibTeX"; }, 2500);
  });
})();
