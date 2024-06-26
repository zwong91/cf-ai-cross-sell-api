import { Hono } from 'hono'
import { env } from 'hono/adapter'
import Stripe from 'stripe'

type Bindings = {
  [key in keyof CloudflareBindings]: CloudflareBindings[key]
}

const app = new Hono<{
  Bindings: Bindings;
  Variables: {
    stripe: Stripe;
  }
}>()

/**
 * Initialize Stripe SDK client
 * We can use this SDK without initializing on each API route,
 * just get it by the following example:
 * ```
 * const stripe = c.get('stripe')
 * ```
 */
app.use('*', async (c, next) => {
  const { STRIPE_SECRET_API_KEY } = env(c)
  const stripe = new Stripe(STRIPE_SECRET_API_KEY)
  c.set('stripe', stripe)
  await next()
})

app.post('/webhook', async (c) => {
  const { STRIPE_WEBHOOK_SECRET } = env(c)
  const stripe = c.get('stripe')
  const signature = c.req.header('stripe-signature')
  if (!signature || !STRIPE_WEBHOOK_SECRET || !stripe) {
      return c.text("", 400)
  }
  try {
    const body = await c.req.text()
    const event = await stripe.webhooks.constructEventAsync(
        body,
        signature,
        STRIPE_WEBHOOK_SECRET,
    )
    if (event.type === 'product.created') {
      const product = event.data.object
      // JSON 格式信息转为 Markdown 格式
      const productData = [
        `## ${product.name}`,
        product.description,
        '### metadata',
        Object.entries(product.metadata).map(([key, value]) => `- ${key}: ${value}`).join('\n')
      ].join('\n')
      console.log(JSON.stringify(productData, null, 2))

      // 调用Text Embedding model worker ai， convert text into vector data of any desired dimension
      const embeddings = await c.env.AI.run(
        '@cf/baai/bge-large-en-v1.5',
        {
          text: productData,
        }
      )
      console.log(JSON.stringify(embeddings, null, 2))

      // Save vector data to Vectorize
      await c.env.VECTORIZE_INDEX.insert([{
        id: product.id,
        values: embeddings.data[0],
        metadata: {
          name: product.name,
          description: product.description || '',
          product_metadata: product.metadata,
        }
      }])
    }
    return c.text("", 200)
  } catch (err) {
    const errorMessage = `⚠️  Webhook signature verification failed. ${err instanceof Error ? err.message : "Internal server error"}`
    console.log(errorMessage)
    return c.text(errorMessage, 400)
  }
})

app.get('/products/:product_id', async (c) => {
  // Get the product ID from API path parameters
  const productId = c.req.param('product_id')

  // Retrieve the indexed data by the product ID
  const [product] = await c.env.VECTORIZE_INDEX.getByIds([productId])

  // Search similar products by using the embedding data
  const similarProducts = await c.env.VECTORIZE_INDEX.query(product.values, {
    topK: 3,
    returnMetadata: true,
    filter: {
      name: {
        "$ne": product.metadata?.name.toString(),
      }
    }
  })

  // TODO：在 Workers KV 或 D1 中保存用户偏好和兴趣类别的信息，然后将该信息作为文本生成的_prompt 使用，可以提供更加准确的推荐功能。

  return c.json({
    product: {
      ...product.metadata
    },
    similarProducts,
  })
})


app.post('/ask', async (c) => {
  const { question } = await c.req.json()
  if (!question) {
    return c.json({
      message: 'Please tell me your question.'
    })
  }
  /**
   * Convert the question to the vector data
   */
  const embeddedQuestion = await c.env.AI.run(
    '@cf/baai/bge-large-en-v1.5',
    {
      text: question,
    }
  )

  /**
   * Query similarity data from Vectorize index
   */
  const similarProducts = await c.env.VECTORIZE_INDEX.query(embeddedQuestion.data[0], {
    topK: 3,
    returnMetadata: true,
  })

  /**
   * Convert the JSON data to the Markdown text
   **/
  const contextData = similarProducts.matches.reduce((prev, current) => {
    if (!current.metadata) return prev
    const productTexts = Object.entries(current.metadata).map(([key, value]) => {
      switch (key) {
        case 'name':
          return `## ${value}`
        case 'product_metadata':
          return `- ${key}: ${JSON.stringify(value)}`
        default:
          return `- ${key}: ${value}`
      }
    })
    const productTextData = productTexts.join('\n')
    return `${prev}\n${productTextData}`
  }, "")

  /**
   * Generate the answer
   */
  const response = await c.env.AI.run(
    "@cf/meta/llama-3-8b-instruct",
    {
      messages: [{
        role: 'system',
        content: `You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.\n#Context: \n${contextData} `
      }, {
        role: 'user',
        content: question
      }]
    }
  )

  return c.json(response)
})

export default app