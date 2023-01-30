function uploadFile(id) {

    const formData = new FormData();
    const uploaded_files = $("#"+id).prop("files")[0];
    
    formData.append("file", uploaded_files);
    formData.append("model_type", id.split("_")[0]);    

    $("#loading").show();
    $.ajax({
        type: "post",
        url: "/upload",
        data: formData,
        contentType: false,
        processData: false,
        success: function (out) {
            console.log(out)
            //show modal --> classification
            // the id of the modal is "classification_modal"
            $("#loading").hide();

            $("#result_modal").show();
            if (out == '1')
            {                
                $("#classification").html('MRI Result: ASD <b>detected</b> <div style="color : crimson ; margin-top : 5px;">Please refer to a specialist.</div>');
            }
            else {
                $("#classification").html('MRI Result: ASD <b>NOT detected</b>');
            }

        },
        error: function (error) {
            console.log(error)
        }
    });
    
}

function showModal(params) {
    
}