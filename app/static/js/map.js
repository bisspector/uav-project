function initMap() {
    map = new google.maps.Map(
        document.getElementById('map'), { zoom: 15, center: uluru, mapTypeId: google.maps.MapTypeId.SATELLITE });
    google.maps.event.addListener(map, 'click', function(e) {
        var location = e.latLng;
        if (state == 0)
            return;
        if (state == 1) {
            console.log(1)
            var marker = new google.maps.Marker({
                position: location,
                map: map,
                icon: iconBase + 'parking_lot_maps.png'
            });
            marker.setMap(map)
            HomeMarker = marker
            sendHomePoint(location)
            state = 2
            return;
        }
        var marker = new google.maps.Marker({
            position: location,
            map: map
        });
        markers.push(marker)
        var lat = marker.getPosition().lat();
        var lng = marker.getPosition().lng();
        points.push({ lat: lat, lng: lng });
        addPoint();
    })
}