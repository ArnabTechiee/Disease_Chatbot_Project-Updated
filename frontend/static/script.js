document.addEventListener("DOMContentLoaded", () => {
    const chatContainer = document.getElementById("chat-container");
    const chatBox = document.getElementById("chat-box");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const closeChatBtn = document.getElementById("close-chat");
    const medicalAssistantBtn = document.getElementById("medical-assistant-btn");
    const overlay = document.getElementById("overlay");
    const logoutBtn = document.getElementById("logout-btn");

    const doctorPopup = document.getElementById("doctor-popup");
    const closePopupBtn = document.getElementById("close-popup");
    const popupOverlay = document.getElementById("popup-overlay");
    const findDoctorBtn = document.querySelector(".sidebar ul li:nth-child(4)"); // Find a Doctor Button

    // ✅ Append messages to chat
    function appendMessage(sender, message, isLoading = false) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add(sender === "user" ? "user-message" : "bot-message");

        if (isLoading) {
            messageDiv.innerHTML = `
                <span class="loading-dots">
                    <span>.</span><span>.</span><span>.</span>
                </span>`;
        } else {
            messageDiv.innerHTML =
                sender === "user"
                    ? `<div class="message-content">${message}</div>`
                    : `<span class="bot-icon">🤖</span> <div class="message-content">${message}</div>`;
        }

        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll to the bottom
        return messageDiv;
    }

    // ✅ Send message
    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        // Display user message
        appendMessage("user", message);
        userInput.value = "";

        // Display loading indicator
        const loadingMessage = appendMessage("bot", "", true);

        try {
            const response = await fetch("/get_response", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const data = await response.json();

            // Remove loading indicator and display bot response
            chatBox.removeChild(loadingMessage);
            appendMessage("bot", data.response || "⚠ No response received.");
        } catch (error) {
            console.error("Error:", error);

            // Remove loading indicator and display error message
            chatBox.removeChild(loadingMessage);
            displayErrorMessage("⚠ Unable to connect. Please try again.");
        }
    }

    // ✅ Display error message
    function displayErrorMessage(errorMessage) {
        appendMessage("bot", errorMessage);
    }

    // ✅ Event Listeners
    sendBtn.addEventListener("click", sendMessage);

    userInput.addEventListener("keypress", (event) => {
        if (event.key === "Enter") sendMessage();
    });

    // ✅ Open chatbot when "Medical Assistant" is clicked
    medicalAssistantBtn.addEventListener("click", () => {
        chatContainer.style.display = "flex";
        overlay.style.display = "block";
        userInput.focus();
    });

    // ✅ Close chatbot when "X" button is clicked
    closeChatBtn.addEventListener("click", closeChat);

    // ✅ Close chatbot when clicking outside the chat container
    overlay.addEventListener("click", closeChat);

    function closeChat() {
        chatContainer.style.display = "none";
        overlay.style.display = "none";
    }

    // ✅ Handle dark mode toggle
    const darkModeToggle = document.getElementById("darkModeToggle");

    darkModeToggle.addEventListener("click", () => {
        document.body.classList.toggle("dark-mode");
        darkModeToggle.innerHTML = document.body.classList.contains("dark-mode")
            ? '<i class="fas fa-sun"></i>'
            : '<i class="fas fa-moon"></i>';
    });

    // ✅ Display notification example
    const notification = document.querySelector(".notification");
    notification.addEventListener("click", () => {
        alert("🔔 You have new notifications!");
    });

    // ✅ Logout Button Handling
    if (logoutBtn) {
        logoutBtn.addEventListener("click", async (event) => {
            event.preventDefault();

            const confirmLogout = confirm("Are you sure you want to log out?");
            if (confirmLogout) {
                try {
                    const response = await fetch("/logout", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                    });

                    if (response.ok) {
                        window.location.href = "/login"; // Redirect to login after logout
                    } else {
                        throw new Error(`Logout failed with status: ${response.status}`);
                    }
                } catch (error) {
                    console.error("Logout Error:", error);
                    displayErrorMessage("⚠ Logout failed. Please try again.");
                }
            }
        });
    }

    // ✅ Open Doctor Consultation Popup
    findDoctorBtn.addEventListener("click", () => {
        doctorPopup.style.display = "block";
        popupOverlay.style.display = "block";
    });

    // ✅ Close Doctor Consultation Popup
    closePopupBtn.addEventListener("click", closeDoctorPopup);
    popupOverlay.addEventListener("click", closeDoctorPopup);

    function closeDoctorPopup() {
        doctorPopup.style.display = "none";
        popupOverlay.style.display = "none";
    }
});
