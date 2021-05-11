import mapboxgl from 'mapbox-gl';
import React from 'react';
import ReactDOM from 'react-dom';
import { Button, Input } from 'antd';

mapboxgl.accessToken =
  Your_token;
class Application extends React.Component {
  mapRef = React.createRef();
  map;
  constructor(props: Props) {
    super(props);
    this.state = { clocker: 0, paused: true, code: '' };

    this.maxTime = 360;
    this.trafficpos = require('./log/lightinfo.json');
    this.roadpos = require('./log/roadinfo.json');
    this.ClickStart = this.ClickStart.bind(this);
    this.ClickPause = this.ClickPause.bind(this);
    this.ClickRestart = this.ClickRestart.bind(this);
    this.ClickReset = this.ClickReset.bind(this);
    this.onChange = this.onChange.bind(this);

    setInterval(() => {
      this.upDate(this.map);
    }, 600);
  }

  ClickStart(event) {
    this.setState({ paused: false });
    event.preventDefault();
  }

  ClickPause(event) {
    this.setState({ paused: true });
    event.preventDefault();
  }

  ClickRestart(event) {
    this.setState({ clocker: 0 });
    event.preventDefault();
  }

  ClickReset(event) {
    console.log(this.state.code);
    var timer = parseInt(this.state.code, 10);
    if (timer >= 0 && timer <= this.maxTime) {
      this.setState({ clocker: timer });
    }
    event.preventDefault();
  }

  onChange(event) {
    this.setState({ code: event.target.value });
  }

  upDate(map) {
    if (this.state.paused === true) return;
    if (map.isStyleLoaded()) {
      var clocker = this.state.clocker;
      if (clocker >= this.maxTime) return;
      this.setState({ clocker: this.state.clocker + 1 });
      var data = require('./log/time' + clocker + '.json');
      var car1 = [];
      var car2 = [];
      var car3 = [];
      var car4 = [];
      var red = [];
      var green = [];
      var yellow = [];
      var roadl1 = [];
      var roadl2 = [];
      var roadl3 = [];
      var roadl4 = [];
      const textArr = [];
      const signalLight = [];
      data[0].forEach(function (item, index, array) {
        if (item[2] === 0) {
          car1.push({
            type: 'Feature',
            geometry: { type: 'Point', coordinates: [item[0], item[1]] },
          });
        } else if (item[2] === 1) {
          car2.push({
            type: 'Feature',
            geometry: { type: 'Point', coordinates: [item[0], item[1]] },
          });
        } else if (item[2] === 2) {
          car3.push({
            type: 'Feature',
            geometry: { type: 'Point', coordinates: [item[0], item[1]] },
          });
        } else if (item[2] === 3) {
          car4.push({
            type: 'Feature',
            geometry: { type: 'Point', coordinates: [item[0], item[1]] },
          });
        }
      });
      var tpos = this.trafficpos;
      var rpos = this.roadpos;
      console.log('p', rpos.length);
      try {
        rpos.forEach((item) => {
        if(item.length >= 3){
            const current = item[2];
            const coordinates = [current[0], current[1]];
              textArr.push({
                type: 'Feature',
                geometry: {
                  type: 'Point',
                  coordinates,
                },
                properties: {
                  description: current[2],
                },
              });
          }
        });
      } catch (error) {
        console.log('error ===> draw road point', error);
      }
      console.log('f', textArr.length);
      data[1].forEach(function (item, index, array) {
        const coordinates = tpos[index];
        signalLight.push({
          type: 'Feature',
          geometry: {
            type: 'Point',
            coordinates,
          },
          properties: {
            description: item,
          },
        });
        if (item === 0) {
          red.push({
            type: 'Feature',
            geometry: { type: 'Point', coordinates },
          });
        } else if (item === 1) {
          green.push({
            type: 'Feature',
            geometry: { type: 'Point', coordinates },
          });
        } else if (item === 2) {
          yellow.push({
            type: 'Feature',
            geometry: { type: 'Point', coordinates },
          });
        }
      });
      data[2].forEach(function (item, index, array) {
        if (item <= 2) {
          roadl1.push({
            type: 'Feature',
            geometry: {
              type: 'LineString',
              coordinates: rpos[index].slice(0,2),
            },
          });
        } else if (item <= 5) {
          roadl2.push({
            type: 'Feature',
            geometry: {
              type: 'LineString',
              coordinates: rpos[index].slice(0,2),
            },
          });
        } else if (item <= 10) {
          roadl3.push({
            type: 'Feature',
            geometry: {
              type: 'LineString',
              coordinates: rpos[index].slice(0,2),
            },
          });
        } else {
          roadl4.push({
            type: 'Feature',
            geometry: {
              type: 'LineString',
              coordinates: rpos[index].slice(0,2),
            },
          });
        }
      });
      var pointsdata = { type: 'FeatureCollection', features: car1 };
      if (map.getSource('cars1')) map.getSource('cars1').setData(pointsdata);
      pointsdata = { type: 'FeatureCollection', features: car2 };
      if (map.getSource('cars2')) map.getSource('cars2').setData(pointsdata);
      pointsdata = { type: 'FeatureCollection', features: car3 };
      if (map.getSource('cars3')) map.getSource('cars3').setData(pointsdata);
      pointsdata = { type: 'FeatureCollection', features: car4 };
      if (map.getSource('cars4')) map.getSource('cars4').setData(pointsdata);
      pointsdata = { type: 'FeatureCollection', features: red };
      if (map.getSource('redlight'))
        map.getSource('redlight').setData(pointsdata);
      pointsdata = { type: 'FeatureCollection', features: green };
      if (map.getSource('greenlight'))
        map.getSource('greenlight').setData(pointsdata);
      pointsdata = { type: 'FeatureCollection', features: yellow };
      if (map.getSource('yellowlight'))
        map.getSource('yellowlight').setData(pointsdata);
      pointsdata = { type: 'FeatureCollection', features: roadl1 };
      if (map.getSource('roadl1')) map.getSource('roadl1').setData(pointsdata);
      pointsdata = { type: 'FeatureCollection', features: roadl2 };
      if (map.getSource('roadl2')) map.getSource('roadl2').setData(pointsdata);
      pointsdata = { type: 'FeatureCollection', features: roadl3 };
      if (map.getSource('roadl3')) map.getSource('roadl3').setData(pointsdata);
      pointsdata = { type: 'FeatureCollection', features: roadl4 };
      if (map.getSource('roadl4')) map.getSource('roadl4').setData(pointsdata);
      pointsdata = { type: 'FeatureCollection', features: signalLight };
      if (map.getSource('signalLight'))
        map.getSource('signalLight').setData(pointsdata);
      pointsdata = { type: 'FeatureCollection', features: textArr };
      if (map.getSource('places')) map.getSource('places').setData(pointsdata);
      console.log(clocker);
    }
  }

  componentDidMount() {
    this.map = new mapboxgl.Map({
      container: this.mapRef.current,
      style: 'mapbox://styles/mapbox/streets-v9',
      center: [115.83444109999999, 28.688112099999998],
      zoom: 12,
    });

    this.map.on('load', function () {
      this.addSource('places', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: [],
        },
      });
      this.addLayer({
        id: 'places',
        type: 'symbol',
        source: 'places',
        layout: {
          'text-field': ['get', 'description'],
          'text-font': ['Open Sans Bold', 'Arial Unicode MS Bold'],
          'text-size': 16,
          'text-transform': 'uppercase',
          'text-letter-spacing': 0.05,
          'text-offset': [0, 1.5],
        },
        paint: {
          'text-color': '#f00',
          'text-halo-color': '#00f',
          'text-halo-width': 2,
        },
      });

      this.addSource('cars1', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: [],
        },
      });
      this.addSource('cars2', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: [],
        },
      });
      this.addSource('cars3', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: [],
        },
      });
      this.addSource('cars4', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: [],
        },
      });
      this.addSource('redlight', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: [],
        },
      });

      this.addSource('greenlight', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: [],
        },
      });

      this.addSource('yellowlight', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: [],
        },
      });

      this.addSource('roadl1', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: [],
        },
      });
      this.addSource('roadl2', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: [],
        },
      });
      this.addSource('roadl3', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: [],
        },
      });
      this.addSource('roadl4', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: [],
        },
      });

      this.addSource('signalLight', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: [],
        },
      });

      this.addLayer({
        id: 'signalLight',
        type: 'symbol',
        source: 'signalLight',
        layout: {
          'text-field': ['get', 'description'],
          'text-font': ['Open Sans Bold', 'Arial Unicode MS Bold'],
          'text-size': 11,
          'text-transform': 'uppercase',
          'text-letter-spacing': 0.05,
          'text-offset': [0, 1.5],
        },
        paint: {
          'text-color': '#202',
          'text-halo-color': '#f00',
          'text-halo-width': 2,
        },
      });

      this.addLayer({
        id: 'roadl4',
        type: 'line',
        source: 'roadl4',
        layout: { 'line-join': 'round', 'line-cap': 'round' },
        paint: { 'line-color': '#00FF00', 'line-width': 2 },
      });
      this.addLayer({
        id: 'roadl3',
        type: 'line',
        source: 'roadl3',
        layout: { 'line-join': 'round', 'line-cap': 'round' },
        paint: { 'line-color': '#FFFF00', 'line-width': 2 },
      });
      this.addLayer({
        id: 'roadl2',
        type: 'line',
        source: 'roadl2',
        layout: { 'line-join': 'round', 'line-cap': 'round' },
        paint: { 'line-color': '#FF0000', 'line-width': 2 },
      });
      this.addLayer({
        id: 'roadl1',
        type: 'line',
        source: 'roadl1',
        layout: { 'line-join': 'round', 'line-cap': 'round' },
        paint: { 'line-color': '#330000', 'line-width': 2 },
      });

      this.addLayer({
        id: 'cars1',
        type: 'circle',
        source: 'cars1',
        paint: { 'circle-radius': 3, 'circle-color': '#0000FF' },
      });
      this.addLayer({
        id: 'cars2',
        type: 'circle',
        source: 'cars2',
        paint: { 'circle-radius': 3, 'circle-color': '#00FFFF' },
      });
      this.addLayer({
        id: 'cars3',
        type: 'circle',
        source: 'cars3',
        paint: { 'circle-radius': 3, 'circle-color': '#008888' },
      });
      this.addLayer({
        id: 'cars4',
        type: 'circle',
        source: 'cars4',
        paint: { 'circle-radius': 3, 'circle-color': '#FF0000' },
      });
      this.addLayer({
        id: 'redlight',
        type: 'circle',
        source: 'redlight',
        paint: { 'circle-radius': 2, 'circle-color': '#FF0000' },
      });
      this.addLayer({
        id: 'greenlight',
        type: 'circle',
        source: 'greenlight',
        paint: { 'circle-radius': 2, 'circle-color': '#00FF00' },
      });
      this.addLayer({
        id: 'yellowlight',
        type: 'circle',
        source: 'yellowlight',
        paint: { 'circle-radius': 2, 'circle-color': '#FFFF00' },
      });
    });
  }

  render() {
    return (
      <div>
        <div className="absolute ml12 mt12 bg-white z1">
          <div>
            <Button type="primary">
              <div>
                <div className="ant-upload-text" onClick={this.ClickStart}>
                  Start
                </div>
              </div>
            </Button>
          </div>
          <div>
            <Button type="primary">
              <div>
                <div className="ant-upload-text" onClick={this.ClickPause}>
                  Pause
                </div>
              </div>
            </Button>
          </div>
          <div>
            <Button type="primary">
              <div>
                <div className="ant-upload-text" onClick={this.ClickRestart}>
                  Restart
                </div>
              </div>
            </Button>
          </div>
          <div>
            Time : {this.state.clocker} / {this.maxTime}
          </div>
          <div>
            <Button type="primary">
              <div>
                <div className="ant-upload-text" onClick={this.ClickReset}>
                  TimeSet
                </div>
              </div>
            </Button>
            <Input
              value={this.state.code}
              onChange={this.onChange}
              size="small"
              style={{ width: 50 }}
            />
          </div>
        </div>
        <div ref={this.mapRef} className="absolute top right left bottom" />
      </div>
    );
  }
}

ReactDOM.render(<Application />, document.getElementById('app'));
