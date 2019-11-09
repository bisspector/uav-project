let points = [];
var uluru = { lat: 49.8643077, lng: 36.4920603 };
var map;
let border, flightPlan;
let markers = []
let HomeMarker;
let iconBase =
    'https://developers.google.com/maps/documentation/javascript/examples/full/images/';
let state = 0;
let droneParametrs = {};
let wayMarkers = []
let kek, path = [],
    cur_point = 0,
    curMarker = 0



let drones = [{
        name: "DJI AGRAS MG-1P RTK",
        battery: 12000,
        perkm: 100,
        speed: 15,
        angle_x: 123,
        angle_y: 123,
        height: 100,
        angular: 150,
        maxTime: 22 * 60,
    },
    {
        name: "DJI Mavic 2 Pro",
        battery: 3850,
        perkm: 100,
        speed: 15,
        angle_x: 123,
        angle_y: 123,
        height: 100,
        angular: 200,
        maxTime: 31 * 60,
    },
    {
        name: "Yuneec Typhoon H Pro RealSense RTF",
        battery: 5400,
        perkm: 100,
        speed: 15,
        angle_x: 123,
        angle_y: 123,
        height: 100,
        angular: 85,
        maxTime: 22 * 60,
    },
    {
        name: "Parrot Bluegrass",
        battery: 6700,
        speed: 15,
        perkm: 100,
        angle_x: 123,
        angle_y: 123,
        height: 100,
        angular: 120,
        maxTime: 25 * 60,
    }
]

function max(a, b) {
    if (a > b)
        return a;
    return b;
}

function drawPos() {
    console.log(cur_point)

    var icon1 = {
        url: "http://maps.google.com/mapfiles/kml/pal4/icon46.png", // url
        scaledSize: new google.maps.Size(30, 25), // scaled size
        origin: new google.maps.Point(0, 0), // origin
        anchor: new google.maps.Point(0, 20) // anchor
    };


    if (cur_point >= path.length)
        clearInterval(kek), cur_point = 0;
    var icon2 = {
        url: "http://maps.google.com/mapfiles/ms/icons/blue-dot.png", // url
        scaledSize: new google.maps.Size(30, 25), // scaled size
        origin: new google.maps.Point(0, 0), // origin
        anchor: new google.maps.Point(0, 32) // anchor
    };
    if (cur_point != 0 && path[cur_point - 1].is) {

        var marker = new google.maps.Marker({
            position: path[cur_point - 1],
            map: map,
            icon: icon1
        });
        marker.setMap(map)
        markers.push(marker)
        wayMarkers.push(marker)
    }
    if (curMarker)
        curMarker.setMap();
    curMarker = new google.maps.Marker({
        position: path[cur_point],
        map: map,
        //icon: icon2
    });
    curMarker.setMap(map);
    markers.push(curMarker);
    wayMarkers.push(curMarker)
    cur_point++;
}

function draw() {
    console.log(path)
    kek = setInterval(drawPos, 400)
    console.log(path)
}