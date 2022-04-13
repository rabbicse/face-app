//! [Recognize face]
async function recognizeFace(formData) {
    return new Promise(function (resolve, reject) {
        try {
            return $.ajax({
                method: "POST",
                // url: "http://localhost:5000/match/v1",
                url: "https://192.168.0.134:5000/match/v1",
                data: formData,
                processData: false,
                contentType: false,
                async: true,
                timeout: 5000,
                mimeType: "multipart/form-data",
                success: function (result) {
                    resolve({ status: 0, response: result });
                },
                error: function (err) {
                    console.log("Error...");
                    console.log(err);
                    reject({ status: 1, message: err });
                }
            });
        } catch (e) {
            console.log("Error when call recognize services. Details: ", e);
            reject({ status: 2, message: e });
        }
    });
}
//! [Recognize face]


// //! [Recognize face]
// async function recognizeFace(formData) {
//     return new Promise(function (resolve) {
//         try {
//             $.ajax({
//                 method: "POST",
//                 url: "https://" + hostname + ":5000/match/v1",
//                 data: formData,
//                 processData: false,
//                 contentType: false,
//                 async: true,
//                 timeout: 3000,
//                 mimeType: "multipart/form-data",
//                 success: function (result) {
//                     console.log(result);
//                     var parsedJson = $.parseJSON(result);
//                     var matchScore = parsedJson["score"];

//                     if (matchScore < 0) {
//                         showMessage("Please look at the camera...");
//                         $("#score").html("Undefined");
//                         $("#matchStatus").html("Not Matched");
//                         resolve(false);
//                         return;
//                     }

//                     var score = (parseFloat(matchScore) * 100).toFixed(2) + '%';
//                     $("#score").html(score);

//                     if (matchScore >= 0.65) {
//                         $("#matchStatus").html("Matched");
//                         showMessage("Matched! Score: " + score);
//                     } else {
//                         $("#matchStatus").html("Not Matched");
//                         showMessage("Not Matched! Score: " + score);
//                     }
//                     resolve(true);
//                 },
//                 error: function (err) {
//                     console.log("Error when call service...");
//                     console.log(err);
//                     var error = err.responseText;
//                     showMessage(error);
//                     resolve(false);
//                 }
//             });
//         } catch (e) {
//             console.log("Error when match face: ", e);
//         }
//     });
// }
// //! [Recognize face]