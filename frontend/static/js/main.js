document.addEventListener("DOMContentLoaded", () => {
    // 1. Overall Quality Slider Update
    const qualSlider = document.getElementById("OverallQual");
    const qualBadge = document.getElementById("qual-badge");
    
    if (qualSlider && qualBadge) {
        qualSlider.addEventListener("input", (e) => {
            const val = e.target.value;
            qualBadge.textContent = `${val} / 10`;
            
            // Subtle dynamic color adjustment for quality badge
            if (val <= 4) {
                qualBadge.style.backgroundColor = "rgba(239, 68, 68, 0.15)";
                qualBadge.style.color = "#ef4444";
            } else if (val >= 8) {
                qualBadge.style.backgroundColor = "rgba(16, 185, 129, 0.15)";
                qualBadge.style.color = "#10b981";
            } else {
                qualBadge.style.backgroundColor = "rgba(59, 130, 246, 0.15)";
                qualBadge.style.color = "#3b82f6";
            }
        });
    }

    // 2. Form Submission Handler
    const form = document.getElementById("prediction-form");
    const submitBtn = document.getElementById("submit-btn");
    const loader = document.getElementById("loader");
    const resultCard = document.getElementById("result-card");
    const resultPlaceholder = document.getElementById("result-placeholder");
    const resultDisplay = document.getElementById("result-display");
    
    // Result fields
    const predPriceEl = document.getElementById("pred-price");
    const diffBadge = document.getElementById("diff-badge");
    const badgeText = document.getElementById("badge-text");
    const nhAvgPriceEl = document.getElementById("nh-avg-price");
    const nhNameEl = document.getElementById("nh-name");
    
    // Feature pills
    const pillArea = document.getElementById("pill-area");
    const pillAge = document.getElementById("pill-age");
    const pillBaths = document.getElementById("pill-baths");
    const pillScore = document.getElementById("pill-score");

    if (form) {
        form.addEventListener("submit", async (e) => {
            e.preventDefault();
            
            // Show loading state
            if (submitBtn && loader) {
                submitBtn.disabled = true;
                loader.style.display = "inline-block";
                submitBtn.querySelector("span").textContent = "Calculating...";
            }

            // Gather values
            const grLivArea = parseInt(document.getElementById("GrLivArea").value);
            const totalBsmtSF = parseInt(document.getElementById("TotalBsmtSF").value);
            const lotArea = parseInt(document.getElementById("LotArea").value);
            const yearBuilt = parseInt(document.getElementById("YearBuilt").value);
            const yearRemodAdd = parseInt(document.getElementById("YearRemodAdd").value);
            const neighborhood = document.getElementById("Neighborhood").value;
            const overallQual = parseInt(document.getElementById("OverallQual").value);
            const kitchenQual = document.getElementById("KitchenQual").value;
            const centralAir = document.getElementById("CentralAir").value;
            const bedroomAbvGr = parseInt(document.getElementById("BedroomAbvGr").value);
            const fullBath = parseInt(document.getElementById("FullBath").value);
            const halfBath = parseInt(document.getElementById("HalfBath").value);
            const garageCars = parseInt(document.getElementById("GarageCars").value);

            // Construct payload
            const payload = {
                GrLivArea: grLivArea,
                TotalBsmtSF: totalBsmtSF,
                LotArea: lotArea,
                YearBuilt: yearBuilt,
                YearRemodAdd: yearRemodAdd,
                Neighborhood: neighborhood,
                OverallQual: overallQual,
                KitchenQual: kitchenQual,
                CentralAir: centralAir,
                BedroomAbvGr: bedroomAbvGr,
                FullBath: fullBath,
                HalfBath: halfBath,
                GarageCars: garageCars
            };

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) {
                    throw new Error("Prediction request failed or server error.");
                }

                const data = await response.json();
                
                // Hide placeholder & Show result panel
                resultCard.classList.remove("empty-state");
                resultPlaceholder.style.display = "none";
                resultDisplay.style.display = "flex";

                // Format price predictions with a count up animation
                const targetPrice = data.predicted_price;
                animateValue(predPriceEl, 0, targetPrice, 1000);

                // Format comparison badge
                const diff = data.percent_diff;
                const nhNameFormatted = document.querySelector(`#Neighborhood option[value="${data.neighborhood}"]`).textContent.split(" (")[0];
                
                diffBadge.className = "comparison-badge"; // Reset classes
                if (diff > 0) {
                    diffBadge.classList.add("higher");
                    badgeText.textContent = `${Math.abs(diff)}% higher than ${nhNameFormatted} avg`;
                } else if (diff < 0) {
                    diffBadge.classList.add("lower");
                    badgeText.textContent = `${Math.abs(diff)}% lower than ${nhNameFormatted} avg`;
                } else {
                    diffBadge.classList.add("lower");
                    badgeText.textContent = `Equal to ${nhNameFormatted} avg`;
                }

                // Fill analysis details
                nhAvgPriceEl.textContent = `$${data.neighborhood_avg.toLocaleString("en-US", {minimumFractionDigits: 2})}`;
                nhNameEl.textContent = nhNameFormatted;

                // Update feature pills in UI
                pillArea.textContent = `${(grLivArea + totalBsmtSF).toLocaleString()} sq ft`;
                
                const yrSold = Math.max(2008, yearBuilt, yearRemodAdd);
                const age = Math.max(0, yrSold - yearBuilt);
                pillAge.textContent = `${age} ${age === 1 ? "Year" : "Years"}`;
                
                const totalBaths = fullBath + (0.5 * halfBath);
                pillBaths.textContent = `${totalBaths} ${totalBaths === 1 ? "Bathroom" : "Bathrooms"}`;
                
                // QualityScore uses default OverallCond of 6 in backend metadata defaults
                pillScore.textContent = `${overallQual * 6} (Qual × Cond)`;

            } catch (err) {
                console.error(err);
                alert("An error occurred during prediction: " + err.message);
            } finally {
                // Restore submit button state
                if (submitBtn && loader) {
                    submitBtn.disabled = false;
                    loader.style.display = "none";
                    submitBtn.querySelector("span").textContent = "Calculate Valuation";
                }
            }
        });
    }

    // Number counting animation helper
    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const currentVal = Math.floor(progress * (end - start) + start);
            obj.innerHTML = currentVal.toLocaleString("en-US");
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }
});
