document.getElementById("reportButton").addEventListener("click", () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (!tabs.length) {
          alert("No active tab found.");
          return;
      }

      const url = tabs[0].url;
      console.log("Reporting phishing site:", url);

      fetch("http://127.0.0.1:5000/report-phishing", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url: url }),
      })
      .then(response => response.json())
      .then(() => alert("Thank you for reporting!"))
      .catch(error => console.error("Error reporting phishing site:", error));
  });
});
