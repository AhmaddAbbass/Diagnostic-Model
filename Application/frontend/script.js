const form = document.getElementById("form");
const output = document.getElementById("output");

form.addEventListener("submit", async e => {
    e.preventDefault();
    output.textContent = "… predicting …";
    try {
        const res = await fetch("/predict", { method: "POST", body: new FormData(form) });
        const data = await res.json();
        output.textContent = data.prediction;
    } catch (err) {
        output.textContent = "❌ Error: " + err;
    }
});