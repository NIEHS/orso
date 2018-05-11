import $ from '$';
import React from 'react';
import ReactDOM from 'react-dom';

import MetaPlot from './MetaPlot';
import Browser from './Browser';
import SmallDataView from './SmallDataView';
import SmallPersonalDataView from './SmallPersonalDataView';
import SmallFavoriteDataView from './SmallFavoriteDataView';
// import SmallRecommendedDataView from './SmallRecommendedDataView';
import SmallUserDataView from './SmallUserDataView';
import IntersectionComparison from './IntersectionComparison';
import PieChart from './PieChart';
import SmallUserView from './SmallUserView';
import SmallRecommendedDataView from './SmallRecommendedDataView';
import PCA from './PCA';
import PCAExplore from './PCAExplore';
import Network from './Network';
import NetworkExplore from './NetworkExplore';
import Dendrogram from './Dendrogram';
import DendrogramExplore from './DendrogramExplore';
import Expression from './Expression';
import GeneScatter from './GeneScatter';
import GeneFoldChange from './GeneFoldChange';
import MetaPlotCarousel from './MetaPlotCarousel';
import GeneValues from './GeneValues';
import RecommendationScatter from './RecommendationScatter';
import ExperimentDataView from './ExperimentDataView';
import DatasetDataView from './DatasetDataView';
import BarChart from './BarChart';


let createMetaPlot = function(el, metaplot) {
    ReactDOM.render(<MetaPlot data={metaplot}/>, el);
};

let createBrowser = function(el, id, assembly, height, width, selectable) {
    ReactDOM.render(<Browser
        id={id}
        assembly={assembly}
        height={height}
        width={width}
        selectable_datasets={selectable}/>, el);
};

let createSmallDataView = function(el, exp_id, meta_data, plot_data, urls, args) {
    ReactDOM.render(<SmallDataView
        exp_id={exp_id}
        meta_data={meta_data}
        plot_data={plot_data}
        urls={urls}
        display_favorite={Boolean(args.display_favorite)}
        display_edit={Boolean(args.display_edit)}
        display_delete={Boolean(args.display_delete)}
        display_remove_recommendation={Boolean(args.display_remove_recommendation)}
        display_remove_favorite={Boolean(args.display_remove_favorite)}/>, el);
};

let appendSmallDataView = function(el, exp_id, meta_data, plot_data, urls, args, recommendation_tags) {
    var element = $('<div></div>').appendTo(el);
    if (recommendation_tags === undefined) recommendation_tags = [];
    ReactDOM.render(<SmallDataView
        exp_id={exp_id}
        meta_data={meta_data}
        plot_data={plot_data}
        urls={urls}
        score={args.score}
        score_dist={args.score_dist}
        display_favorite={Boolean(args.display_favorite)}
        display_edit={Boolean(args.display_edit)}
        display_delete={Boolean(args.display_delete)}
        display_remove_recommendation={Boolean(args.display_remove_recommendation)}
        display_remove_favorite={Boolean(args.display_remove_favorite)}
        recommendation_tags={recommendation_tags}/>, element[0]);
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

// let createSmallRecommendedDataView = function(el, meta_data, promoter_data, enhancer_data, dataset_url, add_favorite_url, remove_favorite_url, hide_recommendation_url) {
//     ReactDOM.render(<SmallRecommendedDataView
//         meta_data={meta_data}
//         promoter_data={promoter_data}
//         enhancer_data={enhancer_data}
//         dataset_url={dataset_url}
//         add_favorite_url={add_favorite_url}
//         remove_favorite_url={remove_favorite_url}
//         hide_recommendation_url={hide_recommendation_url}/>, el);
// };
//
// let appendSmallRecommendedDataView = function(el, meta_data, promoter_data, enhancer_data, dataset_url, add_favorite_url, remove_favorite_url, hide_recommendation_url) {
//     var element = $('<div></div>').appendTo(el);
//
//     ReactDOM.render(<SmallRecommendedDataView
//         meta_data={meta_data}
//         promoter_data={promoter_data}
//         enhancer_data={enhancer_data}
//         dataset_url={dataset_url}
//         add_favorite_url={add_favorite_url}
//         remove_favorite_url={remove_favorite_url}
//         hide_recommendation_url={hide_recommendation_url}/>, element[0]);
// };

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

let createPieChart = function(el, data, index) {
    ReactDOM.render(<PieChart
        data={data}
        id={index}/>, el);
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

let appendSmallRecommendedDataView = function(el, meta_data, plot_data, urls, args) {
    var element = $('<div></div>').appendTo(el);

    ReactDOM.render(<SmallRecommendedDataView
        meta_data={meta_data}
        plot_data={plot_data}
        urls={urls}
        display_favorite={Boolean(args.display_favorite)}
        display_edit={Boolean(args.display_edit)}
        display_delete={Boolean(args.display_delete)}
        display_remove_recommendation={Boolean(args.display_remove_recommendation)}
        display_remove_favorite={Boolean(args.display_remove_favorite)}/>, element[0]);
};

let appendPCA = function(el, pca_data, exp_urls) {
    var element = $('<div></div>').appendTo(el);

    ReactDOM.render(<PCA
        data={pca_data} exp_urls={exp_urls}/>, element[0]);
};

let appendPCAExplore = function(el, pca_lookup, exp_types, assemblies, groups, user_data) {
    var element = $('<div></div>').appendTo(el);

    ReactDOM.render(<PCAExplore
        pca_lookup={pca_lookup}
        available_exp_types={exp_types}
        available_assemblies={assemblies}
        available_groups={groups}
        user_data={user_data}/>, element[0]);
};

let appendNetwork = function(el, network) {
    var element = $('<div></div>').appendTo(el);

    ReactDOM.render(<Network
        network={network}/>, element[0]);
};

let appendNetworkExplore = function(el, network_lookup, available_organisms, available_exp_types) {
    var element = $('<div></div>').appendTo(el);

    ReactDOM.render(<NetworkExplore
        network_lookup={network_lookup}
        available_organisms={available_organisms}
        available_exp_types={available_exp_types}/>, element[0]);
};

let appendDendrogram = function(el, dendrogram) {
    var element = $('<div></div>').appendTo(el);

    ReactDOM.render(<Dendrogram
        dendrogram={dendrogram}/>, element[0]);
};

let appendDendrogramExplore = function(el, dendrogram_lookup, available_organisms, available_exp_types) {
    var element = $('<div></div>').appendTo(el);

    ReactDOM.render(<DendrogramExplore
        dendrogram_lookup={dendrogram_lookup}
        available_organisms={available_organisms}
        available_exp_types={available_exp_types}/>, element[0]);
};

let appendExpression = function(el, data) {
    var element = $('<div></div>').appendTo(el);

    ReactDOM.render(<Expression
        data={data}/>, element[0]);
};

let appendGeneScatter = function(el, data) {
    var element = $('<div></div>').appendTo(el);

    ReactDOM.render(<GeneScatter
        data={data}/>, element[0]);
};

let appendGeneFoldChange = function(el, data) {
    var element = $('<div></div>').appendTo(el);

    ReactDOM.render(<GeneFoldChange
        data={data}/>, element[0]);
};

let appendMetaPlotCarousel = function(el, data) {
    var element = $('<div></div>').appendTo(el);

    ReactDOM.render(<MetaPlotCarousel
        data={data}/>, element[0]);
};

let appendGeneValues = function(el, data) {
    var element = $('<div></div>').appendTo(el);

    ReactDOM.render(<GeneValues
        data={data}/>, element[0]);
};

let appendRecommendationScatter = function(el, data) {
    var element = $('<div></div>').appendTo(el);

    ReactDOM.render(<RecommendationScatter
        data={data}/>, element[0]);
};

let appendExperimentDataView = function(el, data_lookup) {
    var element = $('<div></div>').appendTo(el);

    ReactDOM.render(<ExperimentDataView
        data_lookup={data_lookup}/>, element[0]);
};

let appendDatasetDataView = function(el, data_lookup) {
    var element = $('<div></div>').appendTo(el);

    ReactDOM.render(<DatasetDataView
        data_lookup={data_lookup}/>, element[0]);
};

let createBarChart = function(el, data, index, layout) {
    ReactDOM.render(<BarChart
        data={data}
        id={index}
        layout={layout}/>, el);
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
    // createSmallRecommendedDataView,
    // appendSmallRecommendedDataView,
    createSmallUserDataView,
    appendSmallUserDataView,
    createIntersectionComparison,
    createPieChart,
    appendSmallUserView,
    appendSmallRecommendedDataView,
    appendPCA,
    appendPCAExplore,
    appendNetwork,
    appendNetworkExplore,
    appendDendrogram,
    appendDendrogramExplore,
    appendExpression,
    appendGeneScatter,
    appendGeneFoldChange,
    appendMetaPlotCarousel,
    appendGeneValues,
    appendRecommendationScatter,
    appendExperimentDataView,
    appendDatasetDataView,
    createBarChart,
};
