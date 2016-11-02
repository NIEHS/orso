/*eslint-env node*/

var path = require('path'),
    webpack = require('webpack'),
    BundleTracker = require('webpack-bundle-tracker');


module.exports = {

    context: path.resolve('.'),

    resolve: {
        root: path.resolve('./assets'),
        extensions: ['', '.js', '.css'],
    },

    entry: [
        './assets/index',
    ],

    output: {
        path: path.resolve('./static/bundles'),
        filename: 'bundle.js',
    },

    externals: {
        $: '$',
    },

    plugins: [
        new webpack.optimize.OccurenceOrderPlugin(),
        new webpack.NoErrorsPlugin(),
        new BundleTracker({filename: './webpack-stats.json'}),
    ],

    module: {
        loaders: [
            {
                loaders: [
                    'babel-loader',
                ],
                include: [
                    path.resolve('./assets'),
                ],
                test: /\.js$/,
            },
            {
                test: /\.css$/,
                loader: 'style-loader!css-loader',
            },
        ],
    },

};
