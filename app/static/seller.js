let chartInstance = null;
let globalData = null;

// =========================
// FORMAT CURRENCY (INDIAN)
// =========================
function formatIndianCurrency(num) {
    if (num >= 10000000) {
        return (num / 10000000).toFixed(2) + " Cr";
    } else if (num >= 100000) {
        return (num / 100000).toFixed(2) + " Lakhs";
    } else {
        return num.toLocaleString("en-IN");
    }
}

// =========================
// MAIN PREDICTION
// =========================
async function predictSeller() {
    const location = document.getElementById("location").value;
    const sqft = document.getElementById("sqft").value;
    const bhk = document.getElementById("bhk").value;
    const bought_price = document.getElementById("price").value;

    if (!location || !sqft || !bhk || !bought_price) {
        alert("Fill all fields");
        return;
    }

    const res = await fetch("/api/seller/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            location,
            sqft,
            bhk,
            bought_price
        })
    });

    const data = await res.json();

    if (data.status === "error") {
        alert(data.message);
        return;
    }

    globalData = data;

    // =========================
    // SHOW GROWTH (+ / -)
    // =========================
    const growth = data.growth_rate * 100;
    const growthEl = document.getElementById("growth");

    const sign = growth >= 0 ? "+" : "";
    const isPositiveGrowth = growth >= 0;

    growthEl.innerText = `Expected Growth (1Y): ${sign}${growth.toFixed(2)}%`;
    growthEl.style.color = isPositiveGrowth ? "green" : "red";

    // =========================
    // RENDER CHART
    // =========================
    renderChart(data.predictions);
}

// =========================
// CHART (1Y, 5Y, 10Y)
// =========================
function renderChart(predictions) {
    const ctx = document.getElementById("chart").getContext("2d");

    if (chartInstance) chartInstance.destroy();

    // Trend based on long-term growth
    const isPositiveTrend = predictions["10y"] >= predictions["1y"];
    const lineColor = isPositiveTrend ? "green" : "red";

    chartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: ["1 Year", "5 Years", "10 Years"],
            datasets: [{
                label: "Predicted Price",
                data: [
                    predictions["1y"],
                    predictions["5y"],
                    predictions["10y"]
                ],
                borderColor: lineColor,
                backgroundColor: lineColor,
                borderWidth: 3,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: true
                }
            }
        }
    });
}

// =========================
// BUTTON LOGIC (PRICE + PROFIT)
// =========================
function showPrice(period) {
    if (!globalData) {
        alert("Run prediction first");
        return;
    }

    const price = globalData.predictions[period];
    const bought = parseFloat(document.getElementById("price").value);
    const profit = price - bought;

    const isPositive = profit >= 0;
    const sign = isPositive ? "+" : "-";

    const el = document.getElementById("selectedPrice");

    el.innerText =
        `Bought: ₹ ${formatIndianCurrency(bought)} → After ${period.toUpperCase()}: ₹ ${formatIndianCurrency(price)} 
         (${sign} ₹ ${formatIndianCurrency(Math.abs(profit))})`;

    el.style.color = isPositive ? "green" : "red";
}

async function testConnection() {
    const res = await fetch("/api/test-anylogic");
    const data = await res.json();
    console.log(data);
}