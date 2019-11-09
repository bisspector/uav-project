function clearAll() {
    path = []
    for (var i = 0; i < markers.length; i++)
        if (markers[i])
            markers[i].setMap(null);
    if (border)
        border.setMap(null);
    if (flightPlan)
        flightPlan.setMap(null);
    points = [];
    markers = [];
    state = 2;


    clearInterval(kek);
}

function clearShit() {
    clearAll();
    HomeMarker.setMap(null);
    state = 1;
    clearInterval(kek);
}

function clearWay() {
    for (let i = 0; i < wayMarkers.length; i++)
        wayMarkers[i].setMap(null)
}

function stop() {
    clearInterval(kek);
}

function sendInfo() {
    id = document.getElementById("drone").selectedIndex;
    var angle = 0,
        height = 0,
        ratio = 1,
        overlapping = 0.1,
        maxTime = 0,
        speed = 1,
        angular = 70
    if (id != 0) {
        id--;
        angle_x = drones[id].angle_x;
        angle_y = drones[id].angle_y;
        height = drones[id].height;
        maxTime = drones[id].maxTime;
        speed = drones[id].speed;
        angular = drones[id].angular;
        battery = drones[id].battery;
        perkm = drones[id].perkm;
    }
    if (parseFloat(document.getElementById("angle").value) && parseFloat(document.getElementById("angle").value) > 0) {
        angle_x = parseFloat(document.getElementById("angle").value);
    }
    if (parseFloat(document.getElementById("height").value) && parseFloat(document.getElementById("height").value) > 0) {
        height = parseFloat(document.getElementById("height").value);
    }
    if (parseFloat(document.getElementById("ratio").value) && parseFloat(document.getElementById("ratio").value) > 0) {
        angle_y = parseFloat(document.getElementById("ratio").value);
    }
    if (parseFloat(document.getElementById("speed").value) && parseFloat(document.getElementById("speed").value) > 0) {
        speed = parseFloat(document.getElementById("speed").value)
    }

    if (parseFloat(document.getElementById("battery").value) && parseFloat(document.getElementById("battery").value) > 0) {
        battery = parseFloat(document.getElementById("battery").value)
    }
    if (parseFloat(document.getElementById("perkm").value) && parseFloat(document.getElementById("perkm").value) > 0) {
        perkm = parseFloat(document.getElementById("perkm").value)
    }
    /*if (parseFloat(document.getElementById("angular").value) && parseFloat(document.getElementById("angular").value) > 0) {
        angular = parseFloat(document.getElementById("angular").value)
    }*/
    /*angle = max(0, parseFloat(document.getElementById("angle").value));
    height = max(0, parseFloat(document.getElementById("height").value));
    ratio = max(1, parseFloat(document.getElementById("ratio").value));
    overlapping = max(0.1, parseFloat(document.getElementById("overlapping").value));
    maxTime = max(1000000000, 60 * parseFloat(document.getElementById("maxTime").value));
    speed = max(25, parseFloat(document.getElementById("speed").value));
    angular = max(120, parseFloat(document.getElementById("angular").value));
    angle = 111;
    height = 70;
    ratio = 1;
    overlapping = 0.1;
    maxTime = 11111111;
    speed = 111;
    angular = 120;
    */
    angular = 10000
    drone = { angle_x: angle_x, height: height, angle_y: angle_y, overlapping: overlapping, battery: battery, perkm: perkm, speed: speed, angular: angular };
    if (!speed || speed > 15 || !angle_x || !angle_x > 170 || !height || !overlapping || height > 500 || !angle_y || angle_y > 170 || overlapping > 0.9 || !angular) {
        alert("Please, enter correct info!");
        return;
    }
    var xhr = new XMLHttpRequest();
    var url = "/sendInfo";
    xhr.open("POST", url, true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4 && xhr.status === 200) {
            document.getElementById('solve').classList.remove('inactive');
            document.getElementById('solve').classList.add('active');
            document.getElementById('solve1').classList.remove('inactive');
            document.getElementById('solve1').classList.add('active');
        }
    };
    var data = JSON.stringify(drone);
    xhr.send(data);
    if (state == 0)
        state = 1;
    alert("Data succesfully sent!!");
}

function sendHomePoint(pos) {
    var xhr = new XMLHttpRequest();
    var url = "/sendHomePosition";
    xhr.open("POST", url, true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4 && xhr.status === 200) {}
    };
    var data = JSON.stringify(pos);
    xhr.send(data);
}

function sendDataConstructive() {
    clearWay()
    if (points.size < 3) {
        alert("Please, choose a field")
        return
    }
    var xhr = new XMLHttpRequest();
    var url = "/sendConstructive";
    xhr.open("POST", url, true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4 && xhr.status === 200) {
            var result = JSON.parse(xhr.responseText);
            console.log(result);
            if (flightPlan) {
                flightPlan.setMap(null);
            }
            path = result.way
            flightPlan = new google.maps.Polyline({
                path: result.path,
                geodesic: true,
                strokeColor: '#00FF00',
                strokeOpacity: 1.0,
                strokeWeight: 2
            });
            flightPlan.setMap(map);

            if (!result.ok)
                alert("The charge isn't enough. You need at least: " + (~~(result.needbattery / drone.battery) + 1) + "same charges.")
            else {
                alert("Flight time:" + (~~(result.time / 60)) + "minutes.")
            }
        }
    };
    console.log(points);
    var data = JSON.stringify(points);
    xhr.send(data);
}

function sendDataDroneAlgo() {
    console.log('lool')
    clearWay()
    if (points.size < 3) {
        alert("Please, choose a field.")
        return
    }
    var xhr = new XMLHttpRequest();
    var url = "/sendDroneAlgo";
    xhr.open("POST", url, true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4 && xhr.status === 200) {
            var result = JSON.parse(xhr.responseText);
            console.log(result);
            if (flightPlan) {
                flightPlan.setMap(null);
            }
            path = result.way
            console.log(path)
            flightPlan = new google.maps.Polyline({
                path: result.path,
                geodesic: true,
                strokeColor: '#00FF00',
                strokeOpacity: 1.0,
                strokeWeight: 2
            });
            console.log(result.path);
            flightPlan.setMap(map);
            if (!result.ok)
                alert("The charge isn't enough. You need at least: " + (~~(result.needbattery / drone.battery) + 1) + "same charges.")
            else {
                alert("Flight time:" + (~~(result.time / 60)) + "minutes.")
            }
        }
    };
    console.log(points);
    var data = JSON.stringify(points);
    xhr.send(data);
}


function addPoint() {
    if (border) {
        border.setMap(null);
    }
    points.push(points[0])
    console.log(points)
    border = new google.maps.Polyline({
        path: points,
        geodesic: true,
        strokeColor: '#FF0000',
        strokeOpacity: 1.0,
        strokeWeight: 2
    });
    points.pop()
    border.setMap(map);
}