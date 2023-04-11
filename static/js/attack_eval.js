
var model_py = document.getElementById("model-py");
var model_py_info = document.getElementById("model-py-info");
var model_bin = document.getElementById("model-bin");
var model_bin_info = document.getElementById("model-bin-info")
var train_data = document.getElementById("train-set");
var train_info = document.getElementById("train-info");
var test_data = document.getElementById("test-set");
var test_info = document.getElementById("test-info");
var valid_data = document.getElementById("valid-set");
var valid_info = document.getElementById("valid-info");
var error_info = document.getElementById('error-info');
var upload_datas = new FormData();

model_py.addEventListener("change", function () {

    if (!model_py.value.endsWith("py")) {
        alert("请上传python文件！");
        return;
    }
    var file = model_py.files[0];
    var size = file.size;
    var cnt = 0;
    var unit = "";
    while (size >= 10) {
        size /= 1024;
        cnt++;
    }
    if (cnt == 1)
        unit = "KB";
    if (cnt == 2)
        unit = "MB";
    if (cnt == 3)
        unit = "GB";
    model_py_info.innerHTML = '文件: ' + file.name + '<br>' +
        '大小: ' + parseFloat(size).toFixed(4) + unit + '<br>' +
        '修改: ' + new Date(file.lastModified).toLocaleString();
    upload_datas.append("model_py", file);
    console.log("添加模型结构成功!");
})

model_bin.addEventListener("change", function () {

    if (!model_bin.value.endsWith("bin")) {
        alert("请上传.bin格式的模型文件！");
        return;
    }
    var file = model_bin.files[0];
    var size = file.size;
    var cnt = 0;
    var unit = "";
    while (size >= 10) {
        size /= 1024;
        cnt++;
    }
    if (cnt == 1)
        unit = "KB";
    if (cnt == 2)
        unit = "MB";
    if (cnt == 3)
        unit = "GB";
    model_bin_info.innerHTML = '文件: ' + file.name + '<br>' +
        '大小: ' + parseFloat(size).toFixed(4) + unit + '<br>' +
        '修改: ' + new Date(file.lastModified).toLocaleString();
    upload_datas.append("model_bin", file);
    console.log("添加模型结构成功!");
})

train_data.addEventListener("change", function () {
    // var testing_train_data = $("#train-set").val();
    // console.log("value of set: " + testing_train_data);

    if (!train_data.value.endsWith("jsonl")) {
        alert("请上传.jsonl格式文件！");
        return;
    }
    var file = train_data.files[0];
    var size = file.size;
    var cnt = 0;
    var unit = "";
    while (size >= 10) {
        size /= 1024;
        cnt++;
    }
    if (cnt == 1)
        unit = "KB";
    if (cnt == 2)
        unit = "MB";
    if (cnt == 3)
        unit = "GB";
    train_info.innerHTML = '文件: ' + file.name + '<br>' +
        '大小: ' + parseFloat(size).toFixed(4) + unit + '<br>' +
        '修改: ' + new Date(file.lastModified).toLocaleString();
    upload_datas.append("train_set", file);
    console.log("添加训练集成功!");
})

test_data.addEventListener("change", function () {

    if (!test_data.value.endsWith("jsonl")) {
        alert("请上传.jsonl格式文件！");
        return;
    }
    var file = test_data.files[0];
    var size = file.size;
    var cnt = 0;
    var unit = "";
    while (size >= 10) {
        size /= 1024;
        cnt++;
    }
    if (cnt == 1)
        unit = "KB";
    if (cnt == 2)
        unit = "MB";
    if (cnt == 3)
        unit = "GB";
    test_info.innerHTML = '文件: ' + file.name + '<br>' +
        '大小: ' + parseFloat(size).toFixed(4) + unit + '<br>' +
        '修改: ' + new Date(file.lastModified).toLocaleString();
    upload_datas.append("test_set", file);
    console.log("添加测试集成功!");
})

valid_data.addEventListener("change", function () {

    if (!valid_data.value.endsWith("jsonl")) {
        alert("请上传.jsonl格式文件！");
        return;
    }
    var file = valid_data.files[0];
    var size = file.size;
    var cnt = 0;
    var unit = "";
    while (size >= 10) {
        size /= 1024;
        cnt++;
    }
    if (cnt == 1)
        unit = "KB";
    if (cnt == 2)
        unit = "MB";
    if (cnt == 3)
        unit = "GB";
    valid_info.innerHTML = '文件: ' + file.name + '<br>' +
        '大小: ' + parseFloat(size).toFixed(4) + unit + '<br>' +
        '修改: ' + new Date(file.lastModified).toLocaleString();
    upload_datas.append("valid_set", file);
    console.log("添加验证集成功!");
})


function upload_attack_eval() {
    // for (var key of upload_datas.keys())
    //     console.log(upload_datas.get(key));

    console.log(JSON.stringify(upload_datas));

    if (!model_py.value) {
        alert('没有选择py模型结构文件');
        window.location.reload();
        return;
    }
    if (!model_bin.value) {
        alert('没有.bin模型文件');
        window.location.reload();
        return;
    }
    if (!train_data.value) {
        alert('没有选择训练集');
        window.location.reload();
        return;
    }
    if (!test_data.value) {
        alert('没有选择测试集');
        window.location.reload();
        return;
    }
    if (!valid_data.value) {
        alert('没有选择验证集');
        window.location.reload();
        return;
    }
    alert("正在上传数据");
    $.ajax({
        type: "POST",
        url: "{{ url_for('home.attack_eval') }}",
        contentType: 'application/json;charset=UTF-8',
        data: upload_datas,
        success: function (result) {
            console.log(result);
            if (result.get("code") == '200') {
                alert('数据上传成功');
                window.location.reload();
            } else {
                // alert(result);
                error_info.innerHTML = result.get("message");
            }
        }
    })
}

function reload() {
    window.location.reload();
}