//! [Recognize face]
async function recognizeFace(formData) {
    try {
        return $.ajax({
            method: "POST",
            // url: "http://localhost:5000/match/v1",
            url: "https://192.168.0.134:5000/match/v1",
            data: formData,
            processData: false,
            contentType: false,
            timeout: 5000,
            mimeType: "multipart/form-data",
            success: function (result) {
                console.log(result);
                var parsedJson = $.parseJSON(result);
                var matchScore = parsedJson["score"];

                if(matchScore < 0) {
                    showMessage("Please look at the camera...");
                    $("#score").html("Undefined");
                    $("#matchStatus").html("Not Matched");
                    return;
                }

                var score = (parseFloat(matchScore) * 100).toFixed(2) + '%';
                $("#score").html(score);

                if (matchScore >= 0.65) {
                    $("#matchStatus").html("Matched");
                } else {
                    $("#matchStatus").html("Not Matched");
                }
            },
            error: function (err) {
                console.log("Error...");
                console.log(err);
            }
        });
    } catch (e) {
        console.log(e);
    }
}
//! [Recognize face]