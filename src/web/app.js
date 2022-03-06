const https = require("https");
const express = require('express')
const fs = require('fs');
const path = require('path')

// set port
const port = 3000

const app = express()

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


var key = fs.readFileSync('key.pem');
var cert = fs.readFileSync('cert.pem');
var options = {
  key: key,
  cert: cert
};

var server = https.createServer(options, app);

server.listen(port, () => {
  console.log("server starting on port : " + port)
});

app.get('/facemash', (req, res) => {
  res.sendFile(path.join(__dirname, 'public/html/facemash.html'))
})
app.get('/pose', (req, res) => {
  res.sendFile(path.join(__dirname, 'public/html/pose.html'))
})
app.get('/landmark', (req, res) => {
  res.sendFile(path.join(__dirname, 'public/html/face-lm.html'))
})
app.listen(3001, () => {
  console.log(`Example app listening at http://localhost:${port}`)
})
