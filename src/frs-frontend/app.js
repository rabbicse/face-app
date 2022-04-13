const https = require("https");
const express = require('express')
const fs = require('fs');
const path = require('path')

// set port
const httpPort = 8081;
const httpsPort = 3000;

const app = express()

app.use(express.static('public'))

app.get('/enroll', (req, res) => {
  res.sendFile(path.join(__dirname, 'public/html/enroll.html'))
})

app.get('/match', (req, res) => {
  res.sendFile(path.join(__dirname, 'public/html/match.html'))
})

var key = fs.readFileSync('key.pem');
var cert = fs.readFileSync('cert.pem');
var options = {
  key: key,
  cert: cert
};

var server = https.createServer(options, app);

app.listen(httpPort, () => {
  console.log(`Example app listening at http://localhost:${httpPort}`)
})

server.listen(httpsPort, () => {
  console.log(`Example app listening at https://localhost:${httpsPort}`)
});
