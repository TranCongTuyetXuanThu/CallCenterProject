jQuery(document).ready(function () {
    var $ = jQuery;
    var myRecorder = {
        objects: {
            context: null,
            stream: null,
            recorder: null
        },
        init: function () {
            if (null === myRecorder.objects.context) {
                myRecorder.objects.context = new (
                        window.AudioContext || window.webkitAudioContext
                        );
            }
        },
        start: function () {
            var options = {audio: true, video: false};
            navigator.mediaDevices.getUserMedia(options).then(function (stream) {
                myRecorder.objects.stream = stream;
                myRecorder.objects.recorder = new Recorder(
                        myRecorder.objects.context.createMediaStreamSource(stream),
                        {numChannels: 1}
                );
                myRecorder.objects.recorder.record();
            }).catch(function (err) {});
        },
        stop: function (listObject) {
            if (null !== myRecorder.objects.stream) {
                myRecorder.objects.stream.getAudioTracks()[0].stop();
            }
            if (null !== myRecorder.objects.recorder) {
                myRecorder.objects.recorder.stop();

                // Validate object
                if (null !== listObject
                        && 'object' === typeof listObject
                        && listObject.length > 0) {
                    // Export the WAV file
                    myRecorder.objects.recorder.exportWAV(function (blob) {
                        var url = (window.URL || window.webkitURL)
                                .createObjectURL(blob);

                                var formData = new FormData();
                                formData.append('file', blob, 'recorded_audio.wav');
                        
                                // Send the recorded audio to the server for prediction
                                $.ajax({
                                    type: 'POST',
                                    url: '/predict', // Update the endpoint to match your Flask route
                                    data: formData,
                                    processData: false,
                                    contentType: false,
                                    success: function (response) {
                                        // Handle the prediction response here (e.g., update the UI)
                                        console.log(response);
                                    }
                                });

                        // Prepare the playback
                        var audioObject = $('<audio controls></audio>')
                                .attr('src', url);

                        // Prepare the download link
                        var downloadObject = $('<a>&#9660;</a>')
                                .attr('href', url)
                                .attr('download', new Date().toUTCString() + '.wav');

                        // Wrap everything in a row
                        var holderObject = $('<div class="row"></div>')
                                .append(audioObject)
                                // .append(downloadObject);

                        // Append to the list
                        listObject.append(holderObject);
                    });
                }
            }
        }
    };

    // Prepare the recordings list
    var listObject = $('[data-role="recordings"]');

    // Prepare the record button
    $('[data-role="controls"] > button').click(function () {
        // Initialize the recorder
        myRecorder.init();

        // Get the button state 
        var buttonState = !!$(this).attr('data-recording');

        // Toggle
        if (!buttonState) {
            $(this).attr('data-recording', 'true');
            myRecorder.start();
        } else {
            $(this).attr('data-recording', '');
            myRecorder.stop(listObject);
        }
        
    });
});


    document.getElementById('selectAudioButton').addEventListener('click', function () {
        document.getElementById('audioFile').click();
    });


// document.getElementById('recordButton').addEventListener('click', function () {
//     // Initialize the recorder
//     myRecorder.init();

//     // Get the button state
//     var buttonState = !!$(this).attr('data-recording');

//     // Toggle
//     if (!buttonState) {
//         $(this).attr('data-recording', 'true');
//         myRecorder.start();
//     } else {
//         $(this).attr('data-recording', '');
//         myRecorder.stop(listObject, function () {
//             // After stopping recording, send the recorded data for prediction
//             sendAudioForPrediction(listObject);
//         });
//     }
// });

// function sendAudioForPrediction(listObject) {
//     // Convert the recorded audio data to a WAV file (you may need to adjust this part)
//     var audioData = myRecorder.objects.recorder.exportWAV(function (blob) {
//         var url = (window.URL || window.webkitURL).createObjectURL(blob);

//         // Create a FormData object to send the audio file to the server
//         var formData = new FormData();
//         formData.append('file', blob, 'recorded_audio.wav');

//         // Send the recorded audio to the server for prediction
//         $.ajax({
//             type: 'POST',
//             url: '/predict', // Update the endpoint to match your Flask route
//             data: formData,
//             processData: false,
//             contentType: false,
//             success: function (response) {
//                 // Handle the prediction response here (e.g., update the UI)
//                 console.log(response);
//             }
//         });
//     });
// }