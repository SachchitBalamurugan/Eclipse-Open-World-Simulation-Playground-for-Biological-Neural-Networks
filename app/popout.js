import { initFleet } from './fleet.js';
const id = new URLSearchParams(location.search).get('vm');
document.title = 'NeuroRover Sim - ' + id;
initFleet(document.getElementById('root'), window.lab, { single: id });
