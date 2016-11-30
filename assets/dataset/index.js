import $ from '$';
import React from 'react';
import ReactDOM from 'react-dom';

import MetaPlot from './MetaPlot';
import Browser from './Browser';
import SmallDataView from './SmallDataView';
import SmallPersonalDataView from './SmallPersonalDataView';
import SmallFavoriteDataView from './SmallFavoriteDataView';
import SmallRecommendedDataView from './SmallRecommendedDataView';
import SmallUserDataView from './SmallUserDataView';
import IntersectionComparison from './IntersectionComparison';
import PieChart from './PieChart';
import SmallUserView from './SmallUserView'


let createMetaPlot = function(el, metaplot) {
    ReactDOM.render(<MetaPlot data={metaplot}/>, el);
};

let createBrowser = function(el, id, height, width) {
    ReactDOM.render(<Browser id={id} height={height} width={width}/>, el);
};

let createSmallDataView = function(el, meta_data, plot_data, urls, args) {
    ReactDOM.render(<SmallDataView
        meta_data={meta_data}
        plot_data={plot_data}
        urls={urls}
        display_favorite={Boolean(args.display_favorite)}
        display_edit={Boolean(args.display_edit)}
        display_delete={Boolean(args.display_delete)}
        display_remove_recommendation={Boolean(args.display_remove_recommendation)}
        display_remove_favorite={Boolean(args.display_remove_favorite)}/>, el);
};

let appendSmallDataView = function(el, meta_data, plot_data, urls, args) {
    var element = $('<div></div>').appendTo(el);
    ReactDOM.render(<SmallDataView
        meta_data={meta_data}
        plot_data={plot_data}
        urls={urls}
        display_favorite={Boolean(args.display_favorite)}
        display_edit={Boolean(args.display_edit)}
        display_delete={Boolean(args.display_delete)}
        display_remove_recommendation={Boolean(args.display_remove_recommendation)}
        display_remove_favorite={Boolean(args.display_remove_favorite)}/>, element[0]);
};

let createSmallPersonalDataView = function(el, meta_data, promoter_data, enhancer_data, dataset_url, update_url, delete_url) {
    ReactDOM.render(<SmallPersonalDataView
        meta_data={meta_data}
        promoter_data={promoter_data}
        enhancer_data={enhancer_data}
        dataset_url={dataset_url}
        update_url={update_url}
        delete_url={delete_url}/>, el);
};

let appendSmallPersonalDataView = function(el, meta_data, promoter_data, enhancer_data, dataset_url, update_url, delete_url) {
    var element = $('<div></div>').appendTo(el);
    ReactDOM.render(<SmallPersonalDataView
        meta_data={meta_data}
        promoter_data={promoter_data}
        enhancer_data={enhancer_data}
        dataset_url={dataset_url}
        update_url={update_url}
        delete_url={delete_url}/>, element[0]);
};

let createSmallFavoriteDataView = function(el, meta_data, promoter_data, enhancer_data, dataset_url, remove_favorite_url) {
    ReactDOM.render(<SmallFavoriteDataView
        meta_data={meta_data}
        promoter_data={promoter_data}
        enhancer_data={enhancer_data}
        dataset_url={dataset_url}
        remove_favorite_url={remove_favorite_url}/>, el);
};

let appendSmallFavoriteDataView = function(el, meta_data, promoter_data, enhancer_data, dataset_url, remove_favorite_url) {
    var element = $('<div></div>').appendTo(el);
    ReactDOM.render(<SmallFavoriteDataView
        meta_data={meta_data}
        promoter_data={promoter_data}
        enhancer_data={enhancer_data}
        dataset_url={dataset_url}
        remove_favorite_url={remove_favorite_url}/>, element[0]);
};

let createSmallRecommendedDataView = function(el, meta_data, promoter_data, enhancer_data, dataset_url, add_favorite_url, remove_favorite_url, hide_recommendation_url) {
    ReactDOM.render(<SmallRecommendedDataView
        meta_data={meta_data}
        promoter_data={promoter_data}
        enhancer_data={enhancer_data}
        dataset_url={dataset_url}
        add_favorite_url={add_favorite_url}
        remove_favorite_url={remove_favorite_url}
        hide_recommendation_url={hide_recommendation_url}/>, el);
};

let appendSmallRecommendedDataView = function(el, meta_data, promoter_data, enhancer_data, dataset_url, add_favorite_url, remove_favorite_url, hide_recommendation_url) {
    var element = $('<div></div>').appendTo(el);

    ReactDOM.render(<SmallRecommendedDataView
        meta_data={meta_data}
        promoter_data={promoter_data}
        enhancer_data={enhancer_data}
        dataset_url={dataset_url}
        add_favorite_url={add_favorite_url}
        remove_favorite_url={remove_favorite_url}
        hide_recommendation_url={hide_recommendation_url}/>, element[0]);
};

let createSmallUserDataView = function(el, meta_data, promoter_data, enhancer_data, dataset_url, add_favorite_url, remove_favorite_url, is_favorite) {
    ReactDOM.render(<SmallUserDataView
        meta_data={meta_data}
        promoter_data={promoter_data}
        enhancer_data={enhancer_data}
        dataset_url={dataset_url}
        add_favorite_url={add_favorite_url}
        remove_favorite_url={remove_favorite_url}
        initial_favorite={is_favorite}/>, el);
};

let appendSmallUserDataView = function(el, meta_data, promoter_data, enhancer_data, dataset_url, add_favorite_url, remove_favorite_url, is_favorite) {
    var element = $('<div></div>').appendTo(el);

    ReactDOM.render(<SmallUserDataView
        meta_data={meta_data}
        promoter_data={promoter_data}
        enhancer_data={enhancer_data}
        dataset_url={dataset_url}
        add_favorite_url={add_favorite_url}
        remove_favorite_url={remove_favorite_url}
        initial_favorite={is_favorite}/>, element[0]);
};

let createIntersectionComparison = function(el, x_name, y_name, x_data, y_data) {
    ReactDOM.render(<IntersectionComparison
        x_name={x_name}
        y_name={y_name}
        x_data={x_data}
        y_data={y_data}/>, el);
};

let createPieChart = function(el, data) {
    ReactDOM.render(<PieChart data={data}/>, el);
};

let appendSmallUserView = function(el, meta_data, plot_data, urls, args) {
    var element = $('<div></div>').appendTo(el);

    ReactDOM.render(<SmallUserView
        meta_data={meta_data}
        plot_data={plot_data}
        urls={urls}
        display_favorite={Boolean(args.display_favorite)}
        display_remove_recommendation={Boolean(args.display_remove_recommendation)}
        display_remove_favorite={Boolean(args.display_remove_favorite)}/>, element[0]);
};

export {
    createMetaPlot,
    createBrowser,
    createSmallDataView,
    appendSmallDataView,
    createSmallPersonalDataView,
    appendSmallPersonalDataView,
    createSmallFavoriteDataView,
    appendSmallFavoriteDataView,
    createSmallRecommendedDataView,
    appendSmallRecommendedDataView,
    createSmallUserDataView,
    appendSmallUserDataView,
    createIntersectionComparison,
    createPieChart,
    appendSmallUserView
};
