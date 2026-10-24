const chat = document.querySelector('#chat')
const welcome = document.querySelector('#welcome')
const form = document.querySelector('#composer')
const input = document.querySelector('#message')
const send = document.querySelector('#send')
const clear = document.querySelector('#clear')
const status = document.querySelector('#status')
const statusText = document.querySelector('#status-text')
const pdfInput = document.querySelector('#pdf-input')
const uploadPdf = document.querySelector('#upload-pdf')
const removePdf = document.querySelector('#remove-pdf')
const documentStatus = document.querySelector('#document-status')
const embeddingModel = document.querySelector('#embedding-model')
const embeddingModelStatus = document.querySelector('#embedding-model-status')
let selectedEmbeddingModel = embeddingModel.value
let setupAvailable = false
// This array is the chat's short-term memory. We send it with every request.
let messages = []
let isLoading = false

async function checkSetup() {
  try {
    const response = await fetch('/api/config')
    const config = await response.json()
    if (config.ready) {
      status.classList.add('ready')
      statusText.textContent = `${config.provider} · ${config.model}`
    } else {
      statusText.textContent = 'API key needed'
    }
    setupAvailable = true
    selectedEmbeddingModel = config.embedding_model || selectedEmbeddingModel
    embeddingModel.value = selectedEmbeddingModel
    embeddingModel.disabled = isLoading
    updateDocumentStatus(config.document)
  } catch {
    setupAvailable = false
    embeddingModel.disabled = true
    statusText.textContent = 'Backend offline'
  }
}

function setBusy(busy) {
  isLoading = busy
  send.disabled = busy
  uploadPdf.disabled = busy
  removePdf.disabled = busy
  embeddingModel.disabled = busy || !setupAvailable
}

embeddingModel.addEventListener('change', async () => {
  const requested = embeddingModel.value
  setBusy(true)
  embeddingModelStatus.textContent = 'Loading model and rebuilding document search… Keep this tab open.'
  try {
    const response = await fetch('/api/embedding-model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: requested }),
    })
    const data = await response.json()
    if (!response.ok) throw new Error(data.error || 'The search model could not be changed.')
    selectedEmbeddingModel = data.embedding_model
    embeddingModel.value = selectedEmbeddingModel
    updateDocumentStatus(data.document)
    embeddingModelStatus.textContent = `${embeddingModel.selectedOptions[0].textContent} is ready.`
  } catch (error) {
    embeddingModel.value = selectedEmbeddingModel
    embeddingModelStatus.textContent = `Model unchanged. ${error.message}`
  } finally {
    setBusy(false)
  }
})

function addMessage(role, text, kind = '', sources = []) {
  welcome?.remove()
  const article = document.createElement('article')
  article.className = `message ${role} ${kind}`
  const avatar = document.createElement('div')
  avatar.className = 'avatar'
  if (role === 'user') {
    avatar.textContent = 'You'
  } else {
    const mark = document.createElement('img')
    mark.src = '/brand/contextflow-mark.svg'
    mark.alt = 'ContextFlow'
    mark.className = 'contextflow-avatar'
    avatar.append(mark)
  }
  const content = document.createElement('div')
  content.className = 'message-content'
  const answerText = document.createElement('div')
  answerText.className = 'answer-text'
  renderMarkdown(answerText, text)
  content.append(answerText)
  if (sources.length) {
    const sourceList = document.createElement('details')
    sourceList.className = 'sources'
    const summary = document.createElement('summary')
    summary.textContent = `${sources.length} PDF source${sources.length === 1 ? '' : 's'} used`
    sourceList.append(summary)
    sources.forEach((source) => {
      const item = document.createElement('p')
      if (source.type === 'visual') {
        item.textContent = `[Visual page ${source.page}] ${source.filename} — ${source.text}`
        item.className = 'visual-source'
      } else {
        const pages = source.page_end && source.page_end !== source.page
          ? `pages ${source.page}–${source.page_end}`
          : `page ${source.page}`
        const section = source.section ? `, ${source.section}` : ''
        item.textContent = `[Source ${source.label}] ${source.filename}, ${pages}${section} — ${source.text}`
      }
      sourceList.append(item)
    })
    content.append(sourceList)
  }
  article.append(avatar, content)
  chat.append(article)
  if (text) typesetMath(answerText)
  chat.scrollTop = chat.scrollHeight
  return article
}

function appendInlineMarkdown(parent, text) {
  // Build elements with DOM APIs instead of injecting model-generated HTML.
  const tokenPattern = /(\*\*([^*]+)\*\*|`([^`]+)`|\*([^*]+)\*)/g
  let cursor = 0
  for (const match of text.matchAll(tokenPattern)) {
    parent.append(document.createTextNode(text.slice(cursor, match.index)))
    if (match[2] !== undefined) {
      const strong = document.createElement('strong')
      strong.textContent = match[2]
      parent.append(strong)
    } else if (match[3] !== undefined) {
      const code = document.createElement('code')
      code.textContent = match[3]
      parent.append(code)
    } else {
      const emphasis = document.createElement('em')
      emphasis.textContent = match[4]
      parent.append(emphasis)
    }
    cursor = match.index + match[0].length
  }
  parent.append(document.createTextNode(text.slice(cursor)))
}

function renderMarkdown(container, text) {
  // Some models escape Markdown punctuation even outside code blocks.
  const normalized = text
    .replace(/\\\*\\\*/g, '**')
    .replace(/\\\*([^*\n]+)\\\*/g, '*$1*')
  const lines = normalized.replace(/\r\n/g, '\n').split('\n')
  const isBlockStart = (line) => /^(?:```|#{1,3}\s+|\s*\d+\.\s+|\s*[-+*]\s+)/.test(line)
  let index = 0

  while (index < lines.length) {
    if (!lines[index].trim()) {
      index += 1
      continue
    }

    if (lines[index].trim().startsWith('```')) {
      const language = lines[index].trim().slice(3).trim()
      const codeLines = []
      index += 1
      while (index < lines.length && !lines[index].trim().startsWith('```')) {
        codeLines.push(lines[index])
        index += 1
      }
      if (index < lines.length) index += 1
      const pre = document.createElement('pre')
      const code = document.createElement('code')
      if (language) code.dataset.language = language
      code.textContent = codeLines.join('\n')
      pre.append(code)
      container.append(pre)
      continue
    }

    const heading = lines[index].match(/^(#{1,3})\s+(.+)$/)
    if (heading) {
      const element = document.createElement(`h${heading[1].length + 2}`)
      appendInlineMarkdown(element, heading[2])
      container.append(element)
      index += 1
      continue
    }

    const ordered = lines[index].match(/^\s*\d+\.\s+(.+)$/)
    const unordered = lines[index].match(/^\s*[-+*]\s+(.+)$/)
    if (ordered || unordered) {
      const list = document.createElement(ordered ? 'ol' : 'ul')
      const itemPattern = ordered ? /^\s*\d+\.\s+(.+)$/ : /^\s*[-+*]\s+(.+)$/
      while (index < lines.length) {
        const item = lines[index].match(itemPattern)
        if (!item) break
        const listItem = document.createElement('li')
        appendInlineMarkdown(listItem, item[1])
        list.append(listItem)
        index += 1
      }
      container.append(list)
      continue
    }

    const paragraphLines = []
    while (index < lines.length && lines[index].trim() && (paragraphLines.length === 0 || !isBlockStart(lines[index]))) {
      paragraphLines.push(lines[index].trim())
      index += 1
    }
    const paragraph = document.createElement('p')
    appendInlineMarkdown(paragraph, paragraphLines.join(' '))
    container.append(paragraph)
  }
}

function typesetMath(element) {
  const render = () => {
    if (!window.MathJax?.typesetPromise) return
    window.MathJax.typesetPromise([element]).catch((error) => {
      console.warn('Math rendering failed', error)
    })
  }
  if (window.MathJax?.typesetPromise) {
    render()
  } else {
    document.addEventListener('mathjax-ready', render, { once: true })
  }
}

function addTyping() {
  const article = addMessage('assistant', '')
  article.querySelector('.answer-text').innerHTML = '<div class="typing"><i></i><i></i><i></i></div>'
  return article
}

async function sendMessage(text) {
  if (!text.trim() || isLoading) return null
  const userMessage = { role: 'user', content: text.trim() }
  messages.push(userMessage)
  addMessage('user', userMessage.content)
  input.value = ''
  resizeInput()
  setBusy(true)
  const typing = addTyping()

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: messages.slice(-30) }),
    })
    const data = await response.json()
    typing.remove()
    if (!response.ok) throw new Error(data.error || 'The request failed.')
    messages.push({ role: 'assistant', content: data.reply })
    addMessage('assistant', data.reply, '', data.sources || [])
    return data.reply
  } catch (error) {
    typing.remove()
    addMessage('assistant', error.message, 'error')
    throw error
  } finally {
    setBusy(false)
    input.focus()
  }
}

function updateDocumentStatus(documentInfo = {}) {
  if (documentInfo.loaded) {
    const visualText = documentInfo.visual_pages ? ` · ${documentInfo.visual_pages} visual pages` : ''
    const retrievalText = documentInfo.retrieval ? ` · ${documentInfo.retrieval}` : ''
    documentStatus.textContent = `${documentInfo.filename} · ${documentInfo.pages} pages · ${documentInfo.chunks} chunks${visualText}${retrievalText}`
    documentStatus.title = documentInfo.filename
    removePdf.hidden = false
    uploadPdf.textContent = '↻ Replace PDF'
  } else {
    documentStatus.textContent = 'No document loaded'
    documentStatus.removeAttribute('title')
    removePdf.hidden = true
    uploadPdf.textContent = '＋ Add PDF'
  }
}

async function uploadDocument(file) {
  if (!file) return
  if (file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf')) {
    addMessage('assistant', 'Please choose a PDF file.', 'error')
    return
  }
  if (file.size > 10 * 1024 * 1024) {
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1)
    const proceed = window.confirm(
      `This PDF is ${sizeMb} MB, larger than 10 MB. Uploading and indexing a large book can take several minutes or longer. Keep this tab open until it is ready.\n\nAre you okay with waiting? Continue uploading?`
    )
    if (!proceed) {
      pdfInput.value = ''
      return
    }
  }

  setBusy(true)
  const startedAt = Date.now()
  const updateProcessingStatus = () => {
    const elapsed = Math.floor((Date.now() - startedAt) / 1000)
    const duration = `${Math.floor(elapsed / 60)}m ${elapsed % 60}s`
    documentStatus.textContent = `Uploading and indexing ${file.name}… ${duration}. Keep this tab open.`
  }
  updateProcessingStatus()
  const processingTimer = window.setInterval(updateProcessingStatus, 1000)
  try {
    const response = await fetch('/api/documents', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/pdf',
        'X-Filename': encodeURIComponent(file.name),
      },
      body: file,
    })
    const data = await response.json()
    if (!response.ok) throw new Error(data.error || 'The PDF could not be uploaded.')
    window.clearInterval(processingTimer)
    updateDocumentStatus(data)
    const storageMessage = data.deduplicated
      ? 'It was already indexed, so I reused the saved copy.'
      : 'It was indexed and saved for future uploads.'
    addMessage('assistant', `${data.filename} is ready. ${storageMessage} Ask me a question about it.`)
  } catch (error) {
    window.clearInterval(processingTimer)
    await checkSetup()
    addMessage('assistant', error.message, 'error')
  } finally {
    window.clearInterval(processingTimer)
    setBusy(false)
    pdfInput.value = ''
  }
}

function resizeInput() {
  input.style.height = 'auto'
  input.style.height = `${Math.min(input.scrollHeight, 170)}px`
}

form.addEventListener('submit', (event) => {
  event.preventDefault()
  sendMessage(input.value)
})
input.addEventListener('input', resizeInput)
input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    form.requestSubmit()
  }
})
document.querySelectorAll('[data-prompt]').forEach((button) => {
  button.addEventListener('click', () => sendMessage(button.dataset.prompt))
})
clear.addEventListener('click', () => {
  messages = []
  window.location.reload()
})
uploadPdf.addEventListener('click', () => pdfInput.click())
pdfInput.addEventListener('change', () => uploadDocument(pdfInput.files[0]))
removePdf.addEventListener('click', async () => {
  removePdf.disabled = true
  try {
    const response = await fetch('/api/documents', { method: 'DELETE' })
    const data = await response.json()
    if (!response.ok) throw new Error(data.error || 'The PDF could not be removed.')
    updateDocumentStatus(data)
  } catch (error) {
    addMessage('assistant', error.message, 'error')
  } finally {
    removePdf.disabled = false
  }
})

checkSetup()
input.focus()

// Optional WebMCP support: compatible browsers can use the same visible send
// action as a structured tool. Feature detection keeps normal browsers unchanged.
if (document.modelContext?.registerTool) {
  const registration = document.modelContext.registerTool({
    name: 'send_chat_message',
    title: 'Send chat message',
    description: 'Send one user message in the visible chat and return the assistant reply.',
    inputSchema: {
      type: 'object',
      properties: { message: { type: 'string', minLength: 1, maxLength: 6000 } },
      required: ['message'],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: false, untrustedContentHint: true },
    async execute({ message }) {
      if (typeof message !== 'string' || !message.trim()) throw new Error('message is required')
      const reply = await sendMessage(message)
      return { reply }
    },
  })
  Promise.resolve(registration).catch((error) => console.warn('WebMCP registration failed', error))
}
