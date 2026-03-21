<script lang="ts">
  import { onMount } from "svelte";
  import { models, refreshModels } from "$lib/stores/models";
  import { sendMessage } from "$lib/api/client";

  let selectedModel = $state("");
  let prompt = $state("");
  let systemPrompt = $state("");
  let messages: { role: string; content: string }[] = $state([]);
  let sending = $state(false);

  onMount(refreshModels);

  async function handleSend() {
    if (!prompt.trim() || !selectedModel) return;

    const userMessage = prompt;
    messages = [...messages, { role: "user", content: userMessage }];
    prompt = "";
    sending = true;

    try {
      const response = await sendMessage(
        selectedModel,
        userMessage,
        systemPrompt || undefined,
      );

      const assistantText = response.choices?.[0]?.message?.content ?? "No response";
      messages = [...messages, { role: "assistant", content: assistantText }];
    } catch (e) {
      messages = [...messages, { role: "error", content: String(e) }];
    } finally {
      sending = false;
    }
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }
</script>

<div class="chat-page">
  <div class="header">
    <h1>Chat</h1>
    <select bind:value={selectedModel}>
      <option value="">Select a model</option>
      {#each $models as model}
        <option value={model.name}>{model.name}</option>
      {/each}
    </select>
  </div>

  <div class="messages">
    {#if messages.length === 0}
      <div class="empty">
        <p>Select a model and start chatting.</p>
      </div>
    {/if}

    {#each messages as msg}
      <div class="message {msg.role}">
        <span class="role">{msg.role}</span>
        <div class="content">{msg.content}</div>
      </div>
    {/each}

    {#if sending}
      <div class="message assistant">
        <span class="role">assistant</span>
        <div class="content thinking">Thinking...</div>
      </div>
    {/if}
  </div>

  <div class="input-area">
    <textarea
      bind:value={prompt}
      placeholder="Type a message..."
      rows="3"
      onkeydown={handleKeydown}
      disabled={sending}
    ></textarea>
    <button class="btn btn-primary" onclick={handleSend} disabled={sending || !selectedModel}>
      Send
    </button>
  </div>
</div>

<style>
  .chat-page {
    display: flex;
    flex-direction: column;
    height: calc(100vh - 4rem);
    gap: 1rem;
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
  }

  .header select {
    width: auto;
    min-width: 200px;
  }

  .messages {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    padding: 1rem 0;
  }

  .empty {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--text-secondary);
  }

  .message {
    padding: 0.75rem 1rem;
    border-radius: var(--radius);
  }

  .message.user {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
  }

  .message.assistant {
    background: var(--bg-tertiary);
  }

  .message.error {
    background: var(--bg-secondary);
    border: 1px solid var(--error);
  }

  .role {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .content {
    margin-top: 0.25rem;
    white-space: pre-wrap;
    line-height: 1.5;
  }

  .thinking {
    color: var(--text-secondary);
    font-style: italic;
  }

  .input-area {
    display: flex;
    gap: 0.75rem;
    align-items: flex-end;
  }

  .input-area textarea {
    flex: 1;
    resize: none;
  }
</style>
