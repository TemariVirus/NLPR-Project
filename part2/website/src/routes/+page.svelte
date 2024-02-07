<script lang="ts">
    import { PUBLIC_LEX_API } from "$env/static/public";
    import { v4 as uuidv4 } from "uuid";
    import reload_icon from "$lib/images/reload-icon.webp";

    interface ChatMessage {
        message: string;
        from_user: boolean;
    }

    let chat_history: ChatMessage[] = [];
    let user_text: string = "";
    let session_id: string;
    let sending = false;
    invalidateSession();

    function invalidateSession() {
        chat_history = [];
        user_text = "";
        session_id = uuidv4();
    }

    function scrollToCurrent() {
        const messageContainer = document.getElementById("message-container")!;
        messageContainer.scrollTop = messageContainer.scrollHeight;
    }

    async function sendLex() {
        if (sending) return;
        sending = true;

        chat_history = [
            ...chat_history,
            {
                message: user_text,
                from_user: true,
            },
        ];
        user_text = "";
        setTimeout(scrollToCurrent, 1);

        await fetch(`${PUBLIC_LEX_API}?session_id=${session_id}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/text",
            },
            body: chat_history[chat_history.length - 1].message,
        })
            .then((response) => response.json())
            .then((messages) => {
                chat_history = [
                    ...chat_history,
                    ...messages.map((message: string) => ({
                        message,
                        from_user: false,
                    })),
                ];
            })
            .catch((error) => {
                console.error(error);
                alert(
                    "An error occurred while sending the message. Please try again."
                );
            })
            .finally(() => {
                sending = false;
            });

        scrollToCurrent();
    }
</script>

<head>
    <title>Health Buddy</title>
</head>

<div class="bg-image" />
<div class="main-container">
    <h1>Health Buddy</h1>
    <button class="reload-btn" aria-label="reload" on:click={invalidateSession}>
        <img
            src={reload_icon}
            alt="Reload"
            width="48px"
            height="48px"
            draggable="false"
        />
    </button>

    <div class="message-container multiline" id="message-container">
        {#each chat_history as message}
            {#if message.from_user}
                <div class="user message">{message.message}</div>
            {:else}
                <div class="bot message">{message.message}</div>
            {/if}
        {/each}
    </div>
    <div class="input-container">
        <textarea
            class="message-input"
            rows="2"
            cols="50"
            placeholder="Type your message here..."
            bind:value={user_text}
            on:keydown={(e) => {
                if (!e.shiftKey && e.key === "Enter") {
                    e.preventDefault();
                    sendLex();
                }
            }}
        />
        <button class="send-button" on:click={sendLex}>Send</button>
    </div>
</div>

<style>
    h1 {
        margin: 12px 0;
        font-size: 48px;
    }

    .bg-image {
        height: 100vh;
        width: 100vw;
        position: absolute;
        top: 0px;
        left: 0px;
        z-index: -1;
        background-image: url("/src/lib/images/chat-background.webp");
        background-position: center;
        background-repeat: no-repeat;
        background-size: cover;
        filter: blur(12px) brightness(0.5);
        -webkit-filter: blur(12px) brightness(0.5);
    }

    .main-container {
        position: relative;
        display: flex;
        max-width: 600px;
        margin: 24px auto;
        padding: 24px;
        background-color: #f0f0f0;
        background-color: #f0f0f0dd;
        color: #333;
        border-radius: 16px;
        box-shadow: 0 0 16px rgba(0, 0, 0, 0.5);
        overflow: hidden;
        flex-direction: column;
        align-items: center;
    }

    .reload-btn {
        position: absolute;
        top: 42px;
        right: 36px;
        background: none;
        border: 0;
        cursor: pointer;
    }

    .message-container {
        display: flex;
        flex-direction: column;
        padding: 16px;
        height: 60vh;
        width: 100%;
        max-height: 60vh;
        overflow-y: auto;
    }

    .message-container::-webkit-scrollbar {
        width: 10px;
    }

    .message-container::-webkit-scrollbar-track {
        border-radius: 10px;
        background-color: #e7e7e7;
        border: 1px solid #cacaca;
        box-shadow: inset 0 0 6px rgba(0, 0, 0, 0.3);
    }

    .message-container::-webkit-scrollbar-thumb {
        border-radius: 10px;
        background-color: #a7a7a7;
    }

    .user {
        text-align: right;
        align-self: flex-end;
        background-color: #ffae00;
    }

    .bot {
        text-align: left;
        align-self: flex-start;
        background-color: #2196f3;
    }

    .message {
        width: max-content;
        max-width: calc(100% - 32px);
        margin: 8px;
        padding: 8px;
        word-wrap: break-word;
        color: #fff;
        border-radius: 8px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }

    .input-container {
        display: flex;
        padding: 10px;
        flex-direction: row;
    }

    .message-input {
        flex: 1;
        padding: 8px;
        border: 1px solid #ccc;
        border-radius: 16px;
    }

    .send-button {
        margin-left: 10px;
        padding: 8px 16px;
        background-color: #4caf50;
        color: #fff;
        border: none;
        border-radius: 8px;
        cursor: pointer;
    }
</style>
