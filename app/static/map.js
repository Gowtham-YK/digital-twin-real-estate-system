let map = L.map('map').setView([12.97, 77.59], 12);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
}).addTo(map);

let markers = [];

// Clear old markers
function clearMarkers() {
    markers.forEach(marker => map.removeLayer(marker));
    markers = [];
}

// 🔥 NEW: Render markers instead of heatmap
function renderPoints(points, lat, lon) {
    clearMarkers();

    map.setView([lat, lon], 14);

    // Center marker (main location)
    const centerMarker = L.marker([lat, lon]).addTo(map)
        .bindPopup("📍 Selected Location")
        .openPopup();

    markers.push(centerMarker);

    // Nearby prediction points
    points.forEach(p => {
        const marker = L.circleMarker([p.lat, p.lon], {
            radius: 6,
            color: "blue",
            fillColor: "#30a2ff",
            fillOpacity: 0.8
        }).addTo(map);

        marker.bindPopup(
            `💰 Price: ₹ ${Math.round(p.price).toLocaleString()}`
        );

        markers.push(marker);
    });
}