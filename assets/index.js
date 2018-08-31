import $ from '$';
import React from 'react';
import ReactDOM from 'react-dom';

import MetaPlot from 'network/MetaPlot';
import SmallDataView from 'network/SmallDataView';
import SmallUserView from 'network/SmallUserView';
import PCAExplore from 'network/PCAExplore';
import Network from 'network/Network';
import NetworkExplore from 'network/NetworkExplore';
import DendrogramExplore from 'network/DendrogramExplore';
import ExperimentDataView from 'network/ExperimentDataView';
import DatasetDataView from 'network/DatasetDataView';
import BarChart from 'network/BarChart';

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

let appendDendrogramExplore = function(el, dendrogram_lookup, available_organisms, available_exp_types) {
    var element = $('<div></div>').appendTo(el);

    ReactDOM.render(<DendrogramExplore
        dendrogram_lookup={dendrogram_lookup}
        available_organisms={available_organisms}
        available_exp_types={available_exp_types}/>, element[0]);
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


window.apps = {
    appendSmallDataView,
    appendSmallUserView,
    appendPCAExplore,
    appendExperimentDataView,
    appendDatasetDataView,
    createBarChart,
    appendNetwork,
    appendNetworkExplore,
    appendDendrogramExplore,
};
