// import $ from '$';

import React from 'react';
import ReactDOM from 'react-dom';

// ensure css works
import './test.css';

// assert jQuery is imported correctly and contains bootstrap
// console.assert($ !== undefined);
// console.assert($('body').modal !== undefined);

// ensure react and jsx work as expected
import HelloWorld from './testHelloReact';
ReactDOM.render(
    <HelloWorld />,
    document.getElementById('testReactContainer')
);
