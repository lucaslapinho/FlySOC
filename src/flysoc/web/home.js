'use strict';

const $ = id => document.getElementById(id);
const fmt = value => Number(value).toLocaleString('en-US');

function showError(message){$('error').textContent=message;$('error').hidden=false;}

async function init(){
  try{
    const response=await fetch('/api/status');const status=await response.json();if(!response.ok)throw new Error(status.error||`HTTP ${response.status}`);
    $('connection').textContent='Python engine connected';$('engine-state').textContent='ONLINE';
    $('home-pn').textContent=fmt(status.pn_dim);$('home-kc').textContent=fmt(status.kc_dim);$('home-topk').textContent=`Top-K ${fmt(status.top_k)}`;$('home-memory').textContent=fmt(status.training_alerts);$('home-neurons').textContent=status.connectome?fmt(status.connectome.neurons):'NOT LOADED';
    $('run-caption').textContent=`${status.run} · ${status.representation_backend} · ${fmt(status.test_alerts)} test alerts`;
  }catch(error){$('connection').textContent='Local engine unavailable';$('engine-state').textContent='OFFLINE';showError(error.message);}
}

init();
