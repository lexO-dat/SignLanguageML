const express = require('express');
const bodyParser = require('body-parser');

const app = express();
app.use(bodyParser.json());

let ultimaLetra = ''; // Variable global para almacenar la última letra recibida

// Función para obtener la última letra recibida
function obtenerUltimaLetra() {
    return ultimaLetra;
}

// Ruta PUT para actualizar la última letra
app.put('/letra', function(req, res) {
    const letra = req.body.letra || '';
    ultimaLetra = letra; // Actualiza la última letra recibida
    console.log('Letra recibida:', letra);
    res.send('Letra agregada correctamente');
});

// Ruta GET para mostrar la última letra recibida
app.get('/letra', function(req, res) {
    const letra = obtenerUltimaLetra();
    const paginaHTML = renderizarPagina(letra);
    res.send(paginaHTML);
});

// Función para renderizar la página HTML con la letra recibida
function renderizarPagina(letra) {
    return `
        <html>
        <head>
            <meta http-equiv="refresh" content="1"> <!-- Recarga la página cada 5 segundos -->
        </head>
        <body>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    text-align: center;
                }
                h1 {
                    font-size: 10em;
                    margin: 0;
                }
            </style>
            <h1>${letra}</h1>
        </body>
        </html>`;
}

// Escucha en todas las interfaces en el puerto 3000
app.listen(3000, '192.168.56.1', () => {
    console.log('Servidor HTTP en ejecución en el puerto 3000');
});
