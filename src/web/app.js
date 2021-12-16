const express = require('express')
const path = require('path')

const app = express()
const port = 3000

app.use(express.static('public'))

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public/html/index.html'))
})

app.get('/enroll', (req, res) => {
  res.sendFile(path.join(__dirname, 'public/html/enroll.html'))
})

app.get('/match', (req, res) => {
  res.sendFile(path.join(__dirname, 'public/html/match.html'))
})

app.get('/matchtf', (req, res) => {
  res.sendFile(path.join(__dirname, 'public/html/matchtf.html'))
})

app.get('/facemash', (req, res) => {
  res.sendFile(path.join(__dirname, 'public/html/facemash.html'))
})
app.get('/pose', (req, res) => {
  res.sendFile(path.join(__dirname, 'public/html/pose.html'))
})
app.get('/landmark', (req, res) => {
  res.sendFile(path.join(__dirname, 'public/html/face-lm.html'))
})
app.listen(port, () => {
  console.log(`Example app listening at http://localhost:${port}`)
})
