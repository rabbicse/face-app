function showMessage(msg) {
    $('#status').html(msg);
    toast.show();
}


function initToast() {
    // Toast
    var toastElList = [].slice.call(document.querySelectorAll('.toast'))
    var toastList = toastElList.map(function (toastEl) {
        return new bootstrap.Toast(toastEl)
    });
    return toastList[0];
}