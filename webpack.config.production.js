/*eslint-env node*/

var config = require('./webpack.base.js'),
    webpack = require('webpack');


config.devtool = 'source-map';
config.plugins.unshift.apply(config.plugins, [
    new webpack.DefinePlugin({
        'process.env': {
            'NODE_ENV': JSON.stringify('production'),
        },
    }),
    new webpack.optimize.UglifyJsPlugin({
        compressor: {
            warnings: false,
        },
    }),
]);

module.exports = config;
