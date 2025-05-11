document.addEventListener("DOMContentLoaded", () => {
  // Initialize charts and visualizations
  createDisasterTypesChart()
  createYearlyTrendChart()
  createGlobalDisasterMap()
  createImpactScoreChart()
  createRiskAssessmentChart()

  // Set up event listeners
  setupEventListeners()
})

// Create Disaster Types Chart
function createDisasterTypesChart() {
  const disasterTypes = ["Earthquake", "Flood", "Hurricane", "Wildfire", "Drought", "Tsunami"]
  const disasterCounts = [42, 78, 53, 29, 36, 18]
  const colors = ["#1A5F7A", "#E74C3C", "#FFA500", "#2ECC71", "#9B59B6", "#3498DB"]

  const data = [
    {
      labels: disasterTypes,
      values: disasterCounts,
      type: "pie",
      hole: 0.4,
      marker: {
        colors: colors,
      },
      textinfo: "label+percent",
      insidetextorientation: "radial",
    },
  ]

  const layout = {
    margin: { t: 0, b: 0, l: 0, r: 0 },
    showlegend: false,
    height: 320,
  }

  Plotly.newPlot("disasterTypesChart", data, layout)
}

// Create Yearly Trend Chart
function createYearlyTrendChart() {
  const years = ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]
  const disasterCounts = [156, 172, 189, 201, 215, 228, 243, 256]
  const economicImpact = [12.3, 15.7, 18.2, 22.5, 24.1, 27.8, 30.2, 32.5]

  const trace1 = {
    x: years,
    y: disasterCounts,
    name: "Disaster Count",
    type: "scatter",
    mode: "lines+markers",
    line: {
      color: "#1A5F7A",
      width: 3,
    },
    marker: {
      size: 8,
    },
  }

  const trace2 = {
    x: years,
    y: economicImpact,
    name: "Economic Impact ($B)",
    type: "scatter",
    mode: "lines+markers",
    line: {
      color: "#E74C3C",
      width: 3,
    },
    marker: {
      size: 8,
    },
    yaxis: "y2",
  }

  const data = [trace1, trace2]

  const layout = {
    margin: { t: 10, r: 50, l: 50, b: 40 },
    legend: {
      orientation: "h",
      y: 1.1,
    },
    xaxis: {
      showgrid: false,
    },
    yaxis: {
      title: "Disaster Count",
      titlefont: { color: "#1A5F7A" },
      tickfont: { color: "#1A5F7A" },
    },
    yaxis2: {
      title: "Economic Impact ($B)",
      titlefont: { color: "#E74C3C" },
      tickfont: { color: "#E74C3C" },
      overlaying: "y",
      side: "right",
    },
  }

  Plotly.newPlot("yearlyTrendChart", data, layout)
}

// Create Global Disaster Map
function createGlobalDisasterMap() {
  // Sample data for the map
  const mapData = [
    { lat: 19.4326, lon: -99.1332, type: "Earthquake", magnitude: 7.1, deaths: 370, location: "Mexico City, Mexico" },
    { lat: 18.1096, lon: -77.2975, type: "Hurricane", magnitude: 4, deaths: 45, location: "Jamaica" },
    { lat: 37.7749, lon: -122.4194, type: "Wildfire", magnitude: 3, deaths: 85, location: "California, USA" },
    { lat: 35.6762, lon: 139.6503, type: "Earthquake", magnitude: 6.2, deaths: 12, location: "Tokyo, Japan" },
    { lat: 1.3521, lon: 103.8198, type: "Flood", magnitude: 2, deaths: 5, location: "Singapore" },
    { lat: 55.7558, lon: 37.6173, type: "Extreme Weather", magnitude: 2.5, deaths: 3, location: "Moscow, Russia" },
    { lat: -33.8688, lon: 151.2093, type: "Wildfire", magnitude: 4.5, deaths: 33, location: "Sydney, Australia" },
    { lat: 28.6139, lon: 77.209, type: "Flood", magnitude: 3.8, deaths: 120, location: "Delhi, India" },
    { lat: -1.2921, lon: 36.8219, type: "Drought", magnitude: 4.2, deaths: 230, location: "Nairobi, Kenya" },
    { lat: 41.9028, lon: 12.4964, type: "Extreme Weather", magnitude: 2.1, deaths: 7, location: "Rome, Italy" },
  ]

  // Define colors for different disaster types
  const colors = {
    Earthquake: "#E74C3C",
    Hurricane: "#3498DB",
    Wildfire: "#F39C12",
    Flood: "#2980B9",
    Drought: "#D35400",
    "Extreme Weather": "#8E44AD",
    Tsunami: "#16A085",
    Landslide: "#7F8C8D",
  }

  // Create data for the map
  const data = [
    {
      type: "scattergeo",
      mode: "markers",
      lon: mapData.map((d) => d.lon),
      lat: mapData.map((d) => d.lat),
      text: mapData.map(
        (d) => `<b>${d.type}</b><br>Location: ${d.location}<br>Magnitude: ${d.magnitude}<br>Deaths: ${d.deaths}`,
      ),
      marker: {
        size: mapData.map((d) => Math.max(5, d.magnitude * 3)),
        color: mapData.map((d) => colors[d.type] || "#1A5F7A"),
        line: {
          width: 1,
          color: "white",
        },
        opacity: 0.8,
        sizemode: "diameter",
      },
      name: "Disasters",
      hoverinfo: "text",
    },
  ]

  // Define layout
  const layout = {
    geo: {
      scope: "world",
      showland: true,
      landcolor: "rgb(250, 250, 250)",
      subunitcolor: "rgb(217, 217, 217)",
      countrycolor: "rgb(217, 217, 217)",
      showlakes: true,
      lakecolor: "rgb(255, 255, 255)",
      showsubunits: true,
      showcountries: true,
      resolution: 50,
      projection: {
        type: "equirectangular",
      },
      coastlinewidth: 1,
      countrywidth: 1,
      subunitwidth: 1,
    },
    margin: {
      l: 0,
      r: 0,
      t: 0,
      b: 0,
    },
    height: 400,
  }

  // Create the map
  Plotly.newPlot("globalDisasterMap", data, layout, { responsive: true })
}

// Create Impact Score Chart
function createImpactScoreChart() {
  const data = [
    {
      type: "indicator",
      mode: "gauge+number",
      value: 72,
      title: { text: "Global Impact Score", font: { size: 14 } },
      gauge: {
        axis: { range: [null, 100], tickwidth: 1, tickcolor: "darkblue" },
        bar: { color: "#1A5F7A" },
        bgcolor: "white",
        borderwidth: 2,
        bordercolor: "gray",
        steps: [
          { range: [0, 30], color: "#2ECC71" },
          { range: [30, 70], color: "#F39C12" },
          { range: [70, 100], color: "#E74C3C" },
        ],
        threshold: {
          line: { color: "red", width: 4 },
          thickness: 0.75,
          value: 72,
        },
      },
    },
  ]

  const layout = {
    margin: { t: 25, r: 25, l: 25, b: 25 },
    height: 200,
    font: { color: "#333" },
  }

  Plotly.newPlot("impactScoreChart", data, layout)
}

// Create Risk Assessment Chart
function createRiskAssessmentChart() {
  const regions = ["Southeast Asia", "Western US", "Caribbean", "Central Europe", "East Africa"]
  const riskScores = [85, 78, 82, 45, 90]
  const vulnerabilityScores = [75, 65, 85, 45, 90]

  const trace1 = {
    x: regions,
    y: riskScores,
    name: "Risk Level",
    type: "bar",
    marker: {
      color: "#E74C3C",
      opacity: 0.7,
    },
  }

  const trace2 = {
    x: regions,
    y: vulnerabilityScores,
    name: "Economic Vulnerability",
    type: "bar",
    marker: {
      color: "#3498DB",
      opacity: 0.7,
    },
  }

  const data = [trace1, trace2]

  const layout = {
    barmode: "group",
    margin: { t: 10, r: 10, l: 50, b: 80 },
    legend: {
      orientation: "h",
      y: -0.2,
    },
    yaxis: {
      title: "Score (0-100)",
      range: [0, 100],
    },
    xaxis: {
      tickangle: -45,
    },
  }

  Plotly.newPlot("riskAssessmentChart", data, layout)
}

// Set up event listeners
function setupEventListeners() {
  // Time period selector
  const timePeriodSelector = document.getElementById("timePeriodSelector")
  if (timePeriodSelector) {
    timePeriodSelector.addEventListener("change", function () {
      // Update charts based on selected time period
      updateChartsForTimePeriod(this.value)
    })
  }

  // Region filter
  const regionFilter = document.getElementById("regionFilter")
  if (regionFilter) {
    regionFilter.addEventListener("change", function () {
      // Update charts based on selected region
      updateChartsForRegion(this.value)
    })
  }

  // Disaster type filter
  const disasterTypeFilter = document.getElementById("disasterTypeFilter")
  if (disasterTypeFilter) {
    disasterTypeFilter.addEventListener("change", function () {
      // Update charts based on selected disaster type
      updateChartsForDisasterType(this.value)
    })
  }
}

// Update charts based on time period
function updateChartsForTimePeriod(period) {
  console.log(`Updating charts for time period: ${period}`)
  // This would typically involve fetching new data and redrawing charts
  // For demo purposes, we'll just show an alert
  alert(`Charts would update for time period: ${period}`)
}

// Update charts based on region
function updateChartsForRegion(region) {
  console.log(`Updating charts for region: ${region}`)
  // This would typically involve filtering existing data and redrawing charts
  // For demo purposes, we'll just show an alert
  alert(`Charts would update for region: ${region}`)
}

// Update charts based on disaster type
function updateChartsForDisasterType(type) {
  console.log(`Updating charts for disaster type: ${type}`)
  // This would typically involve filtering existing data and redrawing charts
  // For demo purposes, we'll just show an alert
  alert(`Charts would update for disaster type: ${type}`)
}
