const API = "http://127.0.0.1:5000";

console.log("🔥 JS FILE LOADED");

// =========================
// 🌍 MAP + INIT
// =========================
window.onload = function () {
    console.log("🔥 JS Loaded Successfully");

    let mapDiv = document.getElementById("map");

    if (!mapDiv) {
        console.error("❌ Map div not found");
        return;
    }

    if (typeof L === "undefined") {
        console.error("❌ Leaflet not loaded");
        return;
    }

    // Initialize map
    var map = L.map('map').setView([12.9716, 77.5946], 11);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap'
    }).addTo(map);

    console.log("✅ Map initialized");

    // Load markers
    fetch(API + "/properties")
        .then(res => res.json())
        .then(data => {
            data.forEach(p => {
                if (p.lat && p.lng) {
                    L.marker([p.lat, p.lng]).addTo(map)
                        .bindPopup(`₹ ${Math.round(p.price)}`);
                }
            });
        })
        .catch(err => console.error("❌ Map data error:", err));
};
// =========================
// 🔮 PREDICT
// =========================
function predict() {
    console.log("✅ Predict button clicked");

    let location = document.getElementById("location")?.value;
    let area = document.getElementById("area")?.value;
    let bedrooms = document.getElementById("bedrooms")?.value;
    let bath = document.getElementById("bath")?.value;
    let balcony = document.getElementById("balcony")?.value;

    // Validation
    if (!location || !area || !bedrooms || !bath || !balcony) {
        alert("Please fill all fields!");
        return;
    }

    fetch(API + "/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            location: location,
            area: parseFloat(area),
            bedrooms: parseInt(bedrooms),
            bath: parseFloat(bath),
            balcony: parseFloat(balcony)
        })
    })
    .then(res => res.json())
    .then(data => {
        console.log("API Response:", data);

        if (data.predicted_price) {
            let price = Math.round(data.predicted_price);

            let resultEl = document.getElementById("predictionResult");

            if (resultEl) {
                resultEl.innerText =
                    "💰 Predicted Price: ₹ " + price.toLocaleString();
            }

            // Auto-fill simulation
            let base = document.getElementById("base_price");
            if (base) base.value = price;

            // Auto-fill decision
            let pred = document.getElementById("predicted_price");
            if (pred) pred.value = price;

        } else {
            alert("Error: " + data.error);
        }
    })
    .catch(err => {
        console.error("❌ Predict Error:", err);
        alert("Backend not connected!");
    });
}

// =========================
// ⚡ SIMULATION
// =========================
function simulate() {
    let base = document.getElementById("base_price")?.value;
    let interest = document.getElementById("interest_rate")?.value;
    let demand = document.getElementById("demand_factor")?.value;

    if (!base || !interest) {
        alert("Enter base price and interest rate!");
        return;
    }

    fetch(API + "/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            base_price: parseFloat(base),
            interest_rate: parseFloat(interest),
            demand_factor: parseFloat(demand || 1)
        })
    })
    .then(res => res.json())
    .then(data => {
        let el = document.getElementById("simulationResult");
        if (el) {
            el.innerText =
                "📈 Simulated Price: ₹ " + Math.round(data.simulated_price).toLocaleString() +
                " | Risk: " + data.risk;
        }
    })
    .catch(err => console.error("❌ Simulation Error:", err));
}

// =========================
// 📊 DECISION
// =========================
function decision() {
    let actual = document.getElementById("actual_price")?.value;
    let predicted = document.getElementById("predicted_price")?.value;

    if (!actual || !predicted) {
        alert("Enter both actual and predicted price!");
        return;
    }

    fetch(API + "/decision", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            actual_price: parseFloat(actual),
            predicted_price: parseFloat(predicted)
        })
    })
    .then(res => res.json())
    .then(data => {
        let el = document.getElementById("decisionResult");
        if (el) {
            el.innerText =
                "📊 Decision: " + data.decision +
                " | Appreciation: " + data["appreciation_%"].toFixed(2) + "%";
        }
    })
    .catch(err => console.error("❌ Decision Error:", err));
}