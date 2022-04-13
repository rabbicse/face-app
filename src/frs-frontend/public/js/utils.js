//! [Download file from http to browser path]
async function downloadFileAsync(path, uri) {
    return new Promise(function (resolve) {
        try {
            return $.ajax({
                method: "GET",
                xhrFields: {
                    responseType: 'arraybuffer'
                },
                url: uri,
                timeout: 0,
                async: true,
                success: function (response) {
                    console.log("download success...");

                    let data = new Uint8Array(response);
                    // insert model to index db
                    insertModel(dbName, dbVersion, storeName, { "model_name": path, "model": data });                    

                    resolve(data);
                },
                error: function (err) {
                    console.log("Error when download file...", err);
                    resolve(undefined);
                }
            });
        } catch (ex) {
            console.log(ex);
            resolve(undefined);
        }
    });
}
//! [Download file from http to browser path]


//! [Download file from http to browser path]
async function downloadTextFileAsync(path, uri) {
    return new Promise(function (resolve) {
        try {
            $.ajax({
                method: "GET",
                dataType: "text",
                url: uri,
                timeout: 0,
                async: true,
                success: function (response) {
                    console.log("download success...");

                    resolve(response);
                },
                error: function (err) {
                    console.log("Error when download file...", err);
                    resolve(undefined);
                }
            });
        } catch (ex) {
            console.log(ex);
        }
    });
}
//! [Download file from http to browser path]

//! [Thread Sleep]
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
//! [Thread Sleep]

//! [Convert Canvas to blob]
function getCanvasBlob(canvas) {
    return new Promise(function (resolve, reject) {
        canvas.toBlob(function (blob) {
            resolve(blob)
        }, "image/jpeg", 0.95)
    });
}
//! [Convert Canvas to blob]