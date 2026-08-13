document
  .getElementById("consent-form")
  .addEventListener("submit", function (event) {
    event.preventDefault();

    const checkbox = document.getElementById("consent-checkbox");

    if (checkbox.checked) {
      // Record consent verification in session storage
      sessionStorage.setItem("consent_granted", "true");
      sessionStorage.setItem("consent_timestamp", new Date().toISOString());

      window.location.href = "index.html";
    }
  });
