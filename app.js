const express = require('express')
const path = require('path')

const app = express()
const port = 3000

app.use(express.static('public'))

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public/html/index.html'))
})

app.get('/api/face/v1', (req, res) => {
    res.send("Hello World!")
})

app.listen(port, () => {
  console.log(`Example app listening at http://localhost:${port}`)
})