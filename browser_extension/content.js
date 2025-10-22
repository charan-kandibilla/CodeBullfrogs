console.log("Content script is running!");

let mouseMovements = [];
let clicks = [];
let keystrokes = [];

// Track mouse movements
document.addEventListener("mousemove", (event) => {
    mouseMovements.push({ x: event.clientX, y: event.clientY, timestamp: Date.now() });
});

// Track clicks
document.addEventListener("click", (event) => {
    clicks.push({ x: event.clientX, y: event.clientY, timestamp: Date.now() });
});

// Track keystrokes
document.addEventListener("keydown", (event) => {
    keystrokes.push({ key: event.key, timestamp: Date.now() });
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log("Message received in content script:", message);
    sendResponse({ status: "Message received" });
});

// Function to send data to background script
const sendData = () => {
    if (chrome.runtime?.id) {
        chrome.runtime.sendMessage({ type: "user_interaction", data: { mouseMovements, clicks, keystrokes } }, 
            (response) => {
              if (chrome.runtime.lastError) {
                console.error("Error sending message:", chrome.runtime.lastError.message);
              } else {
                console.log("Response received:", response);
              }
            });

        // Clear stored events after sending
        mouseMovements = [];
        clicks = [];
        keystrokes = [];
    } else {
        console.error("Extension context invalidated");
        clearInterval(intervalId);
    }
};

// Send data every 10 seconds
let intervalId = setInterval(sendData, 10000);
