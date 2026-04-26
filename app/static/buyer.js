// =========================
// BUYER PREDICTION SCRIPT
// =========================

async function predict() {
    try {
        const location = document.getElementById("location").value.trim();
        const sqft = document.getElementById("sqft").value;
        const bhk = document.getElementById("bhk").value;

        // Basic validation
        if (!location || !sqft || !bhk) {
            alert("Please fill all fields");
            return;
        }

        // API call
        const res = await fetch("/api/buyer/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                location: location,
                sqft: sqft,
                bhk: bhk
            })
        });

        const data = await res.json();

        // Handle backend error
        if (data.status === "error") {
            alert(data.message);
            return;
        }

        // Show predicted price
        document.getElementById("price").innerText =
            "Predicted Price: ₹ " +
            Math.round(data.predicted_price).toLocaleString();

        // Render map points
        renderPoints(data.heatmap, data.lat, data.lon);

    } catch (error) {
        console.error("Error:", error);
        alert("Something went wrong!");
    }
}

async function testConnection() {
    const res = await fetch("/api/test-anylogic");
    const data = await res.json();
    console.log(data);
}