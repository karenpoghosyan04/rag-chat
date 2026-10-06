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
const openSettings = document.querySelector('#open-settings')
const closeSettings = document.querySelector('#close-settings')
const settingsDialog = document.querySelector('#settings-dialog')
const settingsForm = document.querySelector('#settings-form')
const settingsError = document.querySelector('#settings-error')
const settingsSummary = document.querySelector('#settings-summary')
const resetSettings = document.querySelector('#reset-settings')

const DEFAULT_SETTINGS = {
  sentences_per_chunk: 4,
  overlap_sentences: 1,
  search_method: 'hybrid',
  max_context_chunks: 4,
  bm25_min_relevance: 0.18,
  embedding_min_similarity: 0.35,
  min_score: 0.35,
  hybrid_candidates: 16,
  rrf_k: 60,
  bm25_weight: 1,
  embedding_weight: 1,
  enable_vision: true,
  max_vision_pages: 2,
  temperature: 0.2,
  max_output_tokens: 1200,
  history_messages: 30,
}

function storedSettings() {
  try {
    return { ...DEFAULT_SETTINGS, ...JSON.parse(localStorage.getItem('rag-settings') || '{}') }
  } catch {
    return { ...DEFAULT_SETTINGS }
  }
}

// This array is the chat's short-term memory. We send it with every request.
let messages = []
let isLoading = false
let settings = storedSettings()
let currentDocument = {}

function setFormSettings(values) {
  for (const [name, value] of Object.entries(values)) {
    const field = settingsForm.elements.namedItem(name)
    if (!field) continue
    if (field.type === 'checkbox') field.checked = Boolean(value)
    else field.value = value
  }
  updateSettingsState()
}

function numericSetting(name) {
  return Number(settingsForm.elements.namedItem(name).value)
}

function readFormSettings() {
  return {
    sentences_per_chunk: numericSetting('sentences_per_chunk'),
    overlap_sentences: numericSetting('overlap_sentences'),
    search_method: settingsForm.elements.namedItem('search_method').value,
    max_context_chunks: numericSetting('max_context_chunks'),
    bm25_min_relevance: numericSetting('bm25_min_relevance'),
    embedding_min_similarity: numericSetting('embedding_min_similarity'),
    min_score: numericSetting('min_score'),
    hybrid_candidates: numericSetting('hybrid_candidates'),
    rrf_k: numericSetting('rrf_k'),
    bm25_weight: numericSetting('bm25_weight'),
    embedding_weight: numericSetting('embedding_weight'),
    enable_vision: settingsForm.elements.namedItem('enable_vision').checked,
    max_vision_pages: numericSetting('max_vision_pages'),
    temperature: numericSetting('temperature'),
    max_output_tokens: numericSetting('max_output_tokens'),
    history_messages: numericSetting('history_messages'),
  }
}

function validateSettings(values) {
  if (values.overlap_sentences >= values.sentences_per_chunk) {
    return 'Overlapping sentences must be smaller than sentences per chunk.'
  }
  if (values.hybrid_candidates < values.max_context_chunks) {
    return 'Candidate pool must be at least as large as sources sent to the model.'
  }
  if (values.search_method === 'hybrid' && values.bm25_weight === 0 && values.embedding_weight === 0) {
    return 'At least one hybrid fusion weight must be greater than zero.'
  }
  return ''
}

function updateSettingsState() {
  const method = settingsForm.elements.namedItem('search_method').value
  const bm25Enabled = method !== 'embedding'
  const embeddingsEnabled = method !== 'bm25'
  settingsForm.elements.namedItem('bm25_min_relevance').disabled = !bm25Enabled
  settingsForm.elements.namedItem('embedding_min_similarity').disabled = !embeddingsEnabled
  settingsForm.elements.namedItem('rrf_k').disabled = method !== 'hybrid'
  settingsForm.elements.namedItem('bm25_weight').disabled = method !== 'hybrid'
  settingsForm.elements.namedItem('embedding_weight').disabled = method !== 'hybrid'
  settingsForm.elements.namedItem('max_vision_pages').disabled = !settingsForm.elements.namedItem('enable_vision').checked
  const label = method === 'hybrid' ? 'Hybrid' : method === 'embedding' ? 'Embeddings' : 'BM25'
  settingsSummary.textContent = `${label} · ${settingsForm.elements.namedItem('sentences_per_chunk').value} sentences/chunk`
}

async function reindexDocument() {
  documentStatus.textContent = `Rebuilding ${currentDocument.filename}…`
  const response = await fetch('/api/documents/reindex', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ settings }),
  })
  const data = await response.json()
  if (!response.ok) throw new Error(data.error || 'The PDF index could not be rebuilt.')
  updateDocumentStatus(data)
}

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
    if (!localStorage.getItem('rag-settings') && config.default_settings) {
      settings = { ...DEFAULT_SETTINGS, ...config.default_settings }
    }
    updateDocumentStatus(config.document)
  } catch {
    statusText.textContent = 'Backend offline'
  }
}

function addMessage(role, text, kind = '', sources = []) {
  welcome?.remove()
  const article = document.createElement('article')
  article.className = `message ${role} ${kind}`
  const avatar = document.createElement('div')
  avatar.className = 'avatar'
  avatar.textContent = role === 'user' ? 'You' : 'AI'
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
  if (role === 'assistant' && text) typesetMath(answerText)
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
  isLoading = true
  send.disabled = true
  const typing = addTyping()

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: messages.slice(-settings.history_messages), settings }),
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
    isLoading = false
    send.disabled = false
    input.focus()
  }
}

function updateDocumentStatus(documentInfo = {}) {
  currentDocument = documentInfo
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
    addMessage('assistant', 'That PDF is larger than the 10 MB limit.', 'error')
    return
  }

  uploadPdf.disabled = true
  documentStatus.textContent = `Reading ${file.name}…`
  try {
    const response = await fetch('/api/documents', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/pdf',
        'X-Filename': encodeURIComponent(file.name),
        'X-RAG-Settings': encodeURIComponent(JSON.stringify(settings)),
      },
      body: file,
    })
    const data = await response.json()
    if (!response.ok) throw new Error(data.error || 'The PDF could not be uploaded.')
    updateDocumentStatus(data)
    addMessage('assistant', `${data.filename} is ready. Ask me a question about it.`)
  } catch (error) {
    await checkSetup()
    addMessage('assistant', error.message, 'error')
  } finally {
    uploadPdf.disabled = false
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

openSettings.addEventListener('click', () => {
  settingsError.hidden = true
  setFormSettings(settings)
  settingsDialog.showModal()
})
closeSettings.addEventListener('click', () => settingsDialog.close())
settingsDialog.addEventListener('click', (event) => {
  if (event.target === settingsDialog) settingsDialog.close()
})
settingsForm.elements.namedItem('search_method').addEventListener('change', updateSettingsState)
settingsForm.elements.namedItem('enable_vision').addEventListener('change', updateSettingsState)
settingsForm.elements.namedItem('sentences_per_chunk').addEventListener('input', updateSettingsState)
resetSettings.addEventListener('click', () => {
  settingsError.hidden = true
  setFormSettings(DEFAULT_SETTINGS)
})
settingsForm.addEventListener('submit', async (event) => {
  event.preventDefault()
  settingsError.hidden = true
  if (!settingsForm.reportValidity()) return
  const nextSettings = readFormSettings()
  const validationError = validateSettings(nextSettings)
  if (validationError) {
    settingsError.textContent = validationError
    settingsError.hidden = false
    return
  }
  const requiresReindex = Boolean(
    currentDocument.loaded
    && (nextSettings.sentences_per_chunk !== settings.sentences_per_chunk
      || nextSettings.overlap_sentences !== settings.overlap_sentences)
  )
  settings = nextSettings
  localStorage.setItem('rag-settings', JSON.stringify(settings))
  const applyButton = settingsForm.querySelector('.primary-button')
  applyButton.disabled = true
  applyButton.textContent = requiresReindex ? 'Rebuilding…' : 'Applying…'
  try {
    if (requiresReindex) await reindexDocument()
    settingsDialog.close()
    if (requiresReindex) addMessage('assistant', 'The PDF was reindexed with the new chunk settings.')
  } catch (error) {
    settingsError.textContent = error.message
    settingsError.hidden = false
  } finally {
    applyButton.disabled = false
    applyButton.textContent = 'Apply settings'
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
