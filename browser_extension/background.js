chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "user_interaction") {
        console.log("Sending data to backend:", message.data);

        fetch("http://127.0.0.1:5000/analyze-behavior", {
            method: "POST",
            body: JSON.stringify(message.data),
            headers: { "Content-Type": "application/json" }
        })
        .then(response => response.json())
        .then(data => console.log("Backend Response:", data))
        .catch(error => console.error("Fetch error:", error));
    }
});

// Handle tab updates to check URLs
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === "complete") {
        const apiUrl = "http://127.0.0.1:5000/analyze-url";
        console.log("Checking URL:", tab.url);

        fetch(apiUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url: tab.url }),
        })
        .then(response => response.json())
        .then(data => {
            console.log("API Response:", data);
            if (data.is_phishing) {
                chrome.notifications.create({
                    type: "basic",
                    iconUrl: "icons/icon48.png",
                    title: "Phishing Alert!",
                    message: `${tab.url} is a phishing site!`,
                });
            }
        })
        .catch(error => console.error("API Error:", error));
    }
});
