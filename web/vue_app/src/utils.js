import * as d3 from 'd3';
import math from 'mathjs';
import distinctColors from 'distinct-colors';

const NUM_COLORS = 20;
export const RANDOM_COLORS = distinctColors({count: NUM_COLORS});

export const FMT = (x, fixed=0, money=false) => {
    var f;
    if (money) {
        f = '$,';
    } else {
        f = ','
    }
  if (x === undefined) {
    return 0;
  } else if (isNaN(x)) {
    return '--';
  } else if (x === 0) {
    return d3.format(f)(x)
  } else {
    //return d3.format('$,')(x) + ',000'
    return d3.format(f)((x).toFixed(fixed))
  }
};

export const toPct = (x, fixed=0) => {
  if (isNaN(x)) {
    return '--';
  }
  return (x * 100).toFixed(fixed) + '%';
};

export const col = (A, i) => {
  let size = math.size(A).valueOf();
  let rowCount = size[0];
  return math.subset(A, math.index(math.range(0, rowCount), i));
};

export function colorLuminance(hex, lum) {
  // Validate hex string
  hex = String(hex).replace(/[^0-9a-f]/gi, "");
  if (hex.length < 6) {
    hex = hex.replace(/(.)/g, '$1$1');
  }
  lum = lum || 0;
  // Convert to decimal and change luminosity
  var rgb = "#",
    c;
  for (var i = 0; i < 3; ++i) {
    c = parseInt(hex.substr(i * 2, 2), 16);
    c = Math.round(Math.min(Math.max(0, c + (c * lum)), 255)).toString(16);
    rgb += ("00" + c).substr(c.length);
  }
  return rgb;
}

export function toClosestOrderOfMagnitude(n) {
  if (n === 0) {
    return 0;
  }
  let order = Math.floor(Math.log(n) / Math.LN10 + 0.000000001);
  let scale = Math.pow(10, order);
  return scale * Math.ceil(n / scale);
}

export function pad(n, width, z) {
  // https://stackoverflow.com/questions/10073699/pad-a-number-with-leading-zeros-in-javascript
  z = z || '0';
  n = n + '';
  return n.length >= width ? n : new Array(width - n.length + 1).join(z) + n;
}

// The download function takes a CSV string, the filename and mimeType as parameters
// Scroll/look down at the bottom of this snippet to see how download is called
// https://stackoverflow.com/questions/14964035/how-to-export-javascript-array-info-to-csv-on-client-side
export function download(content, fileName, mimeType) {
  const a = document.createElement('a');
  mimeType = mimeType || 'application/octet-stream';

  if (navigator.msSaveBlob) { // IE10
    navigator.msSaveBlob(new Blob([content], {
      type: mimeType
    }), fileName);
  } else if (URL && 'download' in a) { //html5 A[download]
    a.href = URL.createObjectURL(new Blob([content], {
      type: mimeType
    }));
    a.setAttribute('download', fileName);
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  } else {
    location.href = 'data:application/octet-stream,' + encodeURIComponent(content); // only this mime type is supported
  }
}

export function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}