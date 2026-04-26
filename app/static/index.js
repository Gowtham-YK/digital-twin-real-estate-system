function login() {
    const user = document.getElementById("username").value;
    const pass = document.getElementById("password").value;

    if (user && pass) {
        document.querySelector(".login-box").style.display = "none";
        document.getElementById("roles").style.display = "block";
    } else {
        alert("Enter username and password");
    }
}

function goBuyer() {
    window.location.href = "/buyer";
}

function goSeller() {
    window.location.href = "/seller";
}