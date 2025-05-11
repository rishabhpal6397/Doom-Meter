document.addEventListener("DOMContentLoaded", () => {
  // Toggle sidebar
  const sidebarCollapse = document.getElementById("sidebarCollapse")
  const sidebar = document.getElementById("sidebar")
  const content = document.getElementById("content")

  if (sidebarCollapse) {
    sidebarCollapse.addEventListener("click", () => {
      sidebar.classList.toggle("active")
      content.classList.toggle("active")
    })
  }

  // Initialize tooltips
  const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
  tooltipTriggerList.map((tooltipTriggerEl) => new bootstrap.Tooltip(tooltipTriggerEl))

  // Initialize popovers
  const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'))
  popoverTriggerList.map((popoverTriggerEl) => new bootstrap.Popover(popoverTriggerEl))

  // Auto-dismiss alerts
  const autoAlerts = document.querySelectorAll(".alert-auto-dismiss")
  autoAlerts.forEach((alert) => {
    setTimeout(() => {
      const bsAlert = new bootstrap.Alert(alert)
      bsAlert.close()
    }, 5000)
  })

  // Add animation to cards
  const cards = document.querySelectorAll(".card")
  cards.forEach((card) => {
    card.classList.add("fade-in")
  })

  // Responsive tables
  const tables = document.querySelectorAll("table")
  tables.forEach((table) => {
    if (!table.parentElement.classList.contains("table-responsive")) {
      const wrapper = document.createElement("div")
      wrapper.classList.add("table-responsive")
      table.parentNode.insertBefore(wrapper, table)
      wrapper.appendChild(table)
    }
  })

  // Add active class to current nav item
  const currentLocation = window.location.pathname
  const navLinks = document.querySelectorAll("#sidebar ul li a")
  navLinks.forEach((link) => {
    if (link.getAttribute("href") === currentLocation) {
      link.parentElement.classList.add("active")
    }
  })

  // Initialize date pickers
  const datePickers = document.querySelectorAll(".datepicker")
  datePickers.forEach((picker) => {
    picker.addEventListener("focus", function () {
      this.type = "date"
    })
    picker.addEventListener("blur", function () {
      if (!this.value) {
        this.type = "text"
      }
    })
  })

  // Confirm delete actions
  const confirmDeletes = document.querySelectorAll(".confirm-delete")
  confirmDeletes.forEach((button) => {
    button.addEventListener("click", (e) => {
      if (!confirm("Are you sure you want to delete this item? This action cannot be undone.")) {
        e.preventDefault()
      }
    })
  })

  // Enhance select elements
  const selects = document.querySelectorAll("select:not(.no-enhance)")
  selects.forEach((select) => {
    select.classList.add("form-select")
  })

  // Add required indicator to form labels
  const requiredInputs = document.querySelectorAll("input[required], select[required], textarea[required]")
  requiredInputs.forEach((input) => {
    const label = document.querySelector(`label[for="${input.id}"]`)
    if (label && !label.querySelector(".required-indicator")) {
      const indicator = document.createElement("span")
      indicator.classList.add("required-indicator", "text-danger", "ms-1")
      indicator.textContent = "*"
      label.appendChild(indicator)
    }
  })

  // Initialize dashboard charts if on dashboard page
  if (window.location.pathname === "/dashboard") {
    initializeDashboardCharts()
  }

  // Initialize prediction form if on prediction page
  if (window.location.pathname === "/prediction") {
    initializePredictionForm()
  }

  // Initialize data explorer if on explorer page
  if (window.location.pathname === "/explorer") {
    initializeDataExplorer()
  }

  // Form validation
  const forms = document.querySelectorAll(".needs-validation")
  Array.from(forms).forEach((form) => {
    form.addEventListener(
      "submit",
      (event) => {
        if (!form.checkValidity()) {
          event.preventDefault()
          event.stopPropagation()
        }
        form.classList.add("was-validated")
      },
      false,
    )
  })

  // Smooth scrolling for anchor links
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      e.preventDefault()
      const targetId = this.getAttribute("href")
      if (targetId === "#") return

      const targetElement = document.querySelector(targetId)
      if (targetElement) {
        targetElement.scrollIntoView({
          behavior: "smooth",
        })
      }
    })
  })

  // Dark mode toggle
  const darkModeToggle = document.getElementById("darkModeToggle")
  if (darkModeToggle) {
    const prefersDarkScheme = window.matchMedia("(prefers-color-scheme: dark)")
    const currentTheme = localStorage.getItem("theme")

    if (currentTheme === "dark" || (!currentTheme && prefersDarkScheme.matches)) {
      document.body.classList.add("dark-mode")
      darkModeToggle.checked = true
    }

    darkModeToggle.addEventListener("change", function () {
      if (this.checked) {
        document.body.classList.add("dark-mode")
        localStorage.setItem("theme", "dark")
      } else {
        document.body.classList.remove("dark-mode")
        localStorage.setItem("theme", "light")
      }
    })
  }

  // Handle form submissions with AJAX
  const ajaxForms = document.querySelectorAll("form.ajax-form")
  ajaxForms.forEach((form) => {
    form.addEventListener("submit", (e) => {
      e.preventDefault()

      const submitButton = form.querySelector('[type="submit"]')
      const originalText = submitButton.innerHTML
      submitButton.disabled = true
      submitButton.innerHTML =
        '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...'

      const formData = new FormData(form)

      fetch(form.action, {
        method: form.method,
        body: formData,
        headers: {
          "X-Requested-With": "XMLHttpRequest",
        },
      })
        .then((response) => response.json())
        .then((data) => {
          if (data.success) {
            // Show success message
            const successAlert = document.createElement("div")
            successAlert.className = "alert alert-success alert-dismissible fade show"
            successAlert.innerHTML = `
                        ${data.message}
                        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                    `
            form.prepend(successAlert)

            // Reset form if needed
            if (data.reset) {
              form.reset()
            }

            // Redirect if needed
            if (data.redirect) {
              setTimeout(() => {
                window.location.href = data.redirect
              }, 1000)
            }
          } else {
            // Show error message
            const errorAlert = document.createElement("div")
            errorAlert.className = "alert alert-danger alert-dismissible fade show"
            errorAlert.innerHTML = `
                        ${data.message}
                        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                    `
            form.prepend(errorAlert)
          }
        })
        .catch((error) => {
          console.error("Error:", error)
          const errorAlert = document.createElement("div")
          errorAlert.className = "alert alert-danger alert-dismissible fade show"
          errorAlert.innerHTML = `
                    An error occurred. Please try again.
                    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                `
          form.prepend(errorAlert)
        })
        .finally(() => {
          submitButton.disabled = false
          submitButton.innerHTML = originalText
        })
    })
  })

  // Initialize date pickers
  const select2Dropdowns = document.querySelectorAll(".select2")
  if (typeof $ !== "undefined" && typeof $.fn.select2 !== "undefined") {
    select2Dropdowns.forEach((select) => {
      $(select).select2({
        theme: "bootstrap-5",
      })
    })
  }

  // Handle real-time data updates
  function setupRealTimeUpdates() {
    const realTimeElements = document.querySelectorAll('[data-realtime="true"]')
    if (realTimeElements.length > 0) {
      setInterval(() => {
        realTimeElements.forEach((element) => {
          const url = element.dataset.url
          if (url) {
            fetch(url)
              .then((response) => response.json())
              .then((data) => {
                if (data.html) {
                  element.innerHTML = data.html
                } else if (data.value) {
                  element.textContent = data.value
                }
              })
              .catch((error) => console.error("Real-time update error:", error))
          }
        })
      }, 30000) // Update every 30 seconds
    }
  }

  setupRealTimeUpdates()
})

// Dashboard charts initialization
function initializeDashboardCharts() {
  // This function will be called if we're on the dashboard page
  console.log("Initializing dashboard charts")

  // Add any additional dashboard-specific JavaScript here
  // These functions are defined in dashboard.js
  if (typeof createDisasterTypesChart === "function") {
    createDisasterTypesChart()
  }

  if (typeof createYearlyTrendChart === "function") {
    createYearlyTrendChart()
  }

  if (typeof createGlobalDisasterMap === "function") {
    createGlobalDisasterMap()
  }

  let createImpactScoreChart
  if (typeof createImpactScoreChart === "function") {
    createImpactScoreChart()
  }

  let createRiskAssessmentChart
  if (typeof createRiskAssessmentChart === "function") {
    createRiskAssessmentChart()
  }
}

// Prediction form initialization
function initializePredictionForm() {
  console.log("Initializing prediction form")

  // Magnitude slider value display
  const magnitudeSlider = document.getElementById("pred_magnitude")
  const magnitudeValue = document.getElementById("magnitude_value")

  if (magnitudeSlider && magnitudeValue) {
    magnitudeSlider.addEventListener("input", function () {
      magnitudeValue.textContent = this.value
    })
  }

  // Form submission loading state
  const predictionForm = document.getElementById("predictionForm")
  const predictBtn = document.getElementById("predictBtn")

  if (predictionForm && predictBtn) {
    predictionForm.addEventListener("submit", () => {
      predictBtn.disabled = true
      predictBtn.innerHTML =
        '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Generating...'
    })
  }

  // Region-Country-State cascading dropdowns
  const regionSelect = document.getElementById("pred_region")
  const countrySelect = document.getElementById("pred_country")
  const stateSelect = document.getElementById("pred_state")

  if (regionSelect && countrySelect && stateSelect) {
    // This function should be defined in the prediction.html template
    let updateCountryOptions
    if (typeof updateCountryOptions === "function") {
      regionSelect.addEventListener("change", updateCountryOptions)
    }

    let updateStateOptions
    if (typeof updateStateOptions === "function") {
      countrySelect.addEventListener("change", updateStateOptions)
    }
  }
}

// Data explorer initialization
function initializeDataExplorer() {
  console.log("Initializing data explorer")

  // Year range slider
  const yearRangeSlider = document.getElementById("year-range")
  const startYearDisplay = document.getElementById("start-year-display")
  const endYearDisplay = document.getElementById("end-year-display")

  if (yearRangeSlider && startYearDisplay && endYearDisplay) {
    yearRangeSlider.addEventListener("change", function () {
      const values = this.value.split(",")
      startYearDisplay.textContent = values[0]
      endYearDisplay.textContent = values[1]
    })
  }

  // Filter form submission
  const filterForm = document.getElementById("explorer-filter-form")
  const filterBtn = document.getElementById("apply-filters-btn")

  if (filterForm && filterBtn) {
    filterForm.addEventListener("submit", (e) => {
      e.preventDefault()
      filterBtn.disabled = true
      filterBtn.innerHTML =
        '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Applying...'

      // Call the filter function (should be defined in explorer.html)
      let applyFilters
      if (typeof applyFilters === "function") {
        applyFilters()
      }

      setTimeout(() => {
        filterBtn.disabled = false
        filterBtn.innerHTML = '<i class="fas fa-filter me-2"></i> Apply Filters'
      }, 1000)
    })
  }
}

// Utility functions
function formatNumber(number, decimals = 0) {
  return number.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })
}

function formatCurrency(amount, currency = "USD", decimals = 0) {
  return amount.toLocaleString("en-US", {
    style: "currency",
    currency: currency,
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })
}

function formatDate(dateString) {
  const date = new Date(dateString)
  return date.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  })
}

function getRandomColor() {
  const letters = "0123456789ABCDEF"
  let color = "#"
  for (let i = 0; i < 6; i++) {
    color += letters[Math.floor(Math.random() * 16)]
  }
  return color
}

function truncateText(text, maxLength) {
  if (text.length <= maxLength) return text
  return text.substr(0, maxLength) + "..."
}

function debounce(func, wait) {
  let timeout
  return function (...args) {
    
    clearTimeout(timeout)
    timeout = setTimeout(() => func.apply(this, args), wait)
  }
}
