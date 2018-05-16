import React from 'react';


class PCA extends React.Component {

    constructor(props) {
        super(props);

        var color_by_choices = ['None'];
        color_by_choices = color_by_choices.concat(this.props.plot['color_choices'])

        var hide_marker_keys = [];
        var points_to_visibility = {};

        if (this.props.user_data.length > 0) {
            points_to_visibility['User'] = true;
            hide_marker_keys.push('User');
        }

        var points_keys = Object.keys(this.props.plot['points']);
        for (var i = 0; i < points_keys.length; i++) {
            var key = points_keys[i];

            points_to_visibility[key] = true;

            var number_of_points =this.props.plot['points'][key].length
            if (number_of_points > 0 && key != 'Other') {
                hide_marker_keys.push(key);
            }
        }

        this.state = {
            color_by: 'None',
            color_by_choices: color_by_choices,
            hide_marker_keys: hide_marker_keys,
            points_to_visibility: points_to_visibility,
            points_to_trace_index: {},  // set during initial plotting
            vector_to_trace_index: {},  // set during initial plotting
            vector_visibility: false,
        };
    }

    datasetUrl(dataset_pk) {
        return `/network/dataset/${dataset_pk}/`;
    }

    experimentUrl(experiment_pk) {
        return `/network/experiment/${experiment_pk}/`;
    }

    drawPlotlyPCA() {
        var data = [];  // contains point and vector traces
        var annotations = [];  // annotations to be used in layout

        var points_keys = Object.keys(this.props.plot['points']);
        var vector_keys = Object.keys(this.props.plot['vectors']);

        var points_data = {};
        for (var i = 0; i < points_keys.length; i++) {
            var key = points_keys[i];
            points_data[key] = this.props.plot['points'][key];
        }

        points_keys.push('User');
        points_data['User'] = this.props.user_data;

        var points_to_trace_index = {};
        var vector_to_trace_index = {};

        // Plot points
        for (var i = 0; i < points_keys.length; i++) {

            var key = points_keys[i];
            var point_data = points_data[key];

            var x = [], y = [], z = [], colors = [], names = [];
            for (var j = 0; j < point_data.length; j++) {
                var point = point_data[j];
                x.push(point['transformed_values'][0]);
                y.push(point['transformed_values'][1]);
                z.push(point['transformed_values'][2]);
                if (point['experiment_target'] == '') {
                    names.push(`Cell type: ${point['experiment_cell_type']}`);
                } else {
                    names.push(`Cell type: ${point['experiment_cell_type']}
                               <br>Target: ${point['experiment_target']}`);
                }
                if (point['colors'].hasOwnProperty(this.state.color_by)) {
                    colors.push(point['colors'][this.state.color_by]);
                } else if (point['colors'].hasOwnProperty('Default')) {
                    colors.push(point['colors']['Default']);
                } else {
                    colors.push('#A9A9A9');
                }
            }

            points_to_trace_index[key] = i;

            data.push({
                x: x,
                y: y,
                z: z,
                mode: 'markers',
                text: names,
                point_data: point_data,
                hoverinfo: 'text',
                marker: {
                    size: 12,
                    color: colors,
                    opacity: 0.8,
                },
                type: 'scatter3d',
            });
        }

        this.setState({points_to_trace_index});

        // Plot vectors
        for (var i = 0; i < vector_keys.length; i++) {
            var key = vector_keys[i];
            var vector = this.props.plot['vectors'][key];

            vector_to_trace_index[key] = points_keys.length + i;

            data.push({
                x: [0, vector['point'][0]],
                y: [0, vector['point'][1]],
                z: [0, vector['point'][2]],
                type: 'scatter3d',
                mode: 'lines+text',
                line: {
                    width: 6,
                    color: vector['color'],
                },
                text: ['', vector['label']],
                visible: false,
            });
        }

        this.setState({vector_to_trace_index});

        var layout = {
            showlegend: false,
            // height: 800,
            margin: {
                l: 0,
                r: 0,
                b: 0,
                t: 0,
            },
            scene: {
        		xaxis:{
                    title: '',
                    showticklabels: false,
                },
        		yaxis:{
                    title: '',
                    showticklabels: false,
                },
        		zaxis:{
                    title: '',
                    showticklabels: false,
                },
    		},
        };

        var options = {
            displaylogo: false,
            displayModeBar: false,
            modeBarButtonsToRemove: [
                'sendDataToCloud',
                'resetCameraLastSave3d',
                'hoverClosest3d',
            ],
        };

        Plotly.newPlot('plot', data, layout, options);

        var plot = document.getElementById('plot');
        plot.on('plotly_click', function(data){
            for(var i = 0; i < data.points.length; i++){
                var index = data.points[i].pointNumber;
                var selected_object = data.points[i].data.point_data[index];
                this.setState({'selected_object': selected_object}, this.setSelection);
            }
        }.bind(this));
    }

    drawPlotlyVariance(){

        var data = [{
            y: ['PC 1', 'PC 2', 'PC 3'].reverse(),
            x: this.props.explained_variance.reverse(),
            type: 'bar',
            orientation: 'h',
        }];

        var layout = {
            margin: {
                l: 50,
                r: 50,
                b: 50,
                t: 50,
                pad: 4,
            },
        };

        var options = {
            displayModeBar: false,
        };

        Plotly.newPlot('variance_plot', data, layout, options);
    }

    drawPlotlyComponent(component, div){
        var labels = [],
            values = [];
        for(var i = 0; i < component.length; i++){
            labels.push(component[i][0]);
            values.push(component[i][1]);
        }

        var data = [{
            y: labels.reverse(),
            x: values.reverse(),
            type: 'bar',
            orientation: 'h',
        }];

        var layout = {
            yaxis: {
                tickfont: {
                    size: 10,
                },
            },
            autosize: false,
            width: $('#tabs').width(),
            height: $('#tabs').height(),
            margin: {
                l: 50,
                r: 50,
                b: 50,
                t: 50,
                pad: 4,
            },
        };

        Plotly.newPlot(div, data, layout);
    }

    removePlotly(divElement){
        $(divElement).empty();
    }

    setSelection(){
        $(this.refs.selection_exp_name).text(this.state.selected_object.experiment_name);
        $(this.refs.selection_ds_name).text(this.state.selected_object.dataset_name);
        $(this.refs.selection_cell_type).text(this.state.selected_object.experiment_cell_type);

        if (this.state.selected_object.experiment_target == '') {
            $(this.refs.selection_target).text('--');
        } else {
            $(this.refs.selection_target).text(this.state.selected_object.experiment_target);
        }

        $(this.refs.experiment_link).attr('href', this.experimentUrl(this.state.selected_object.experiment_pk));
        $(this.refs.dataset_link).attr('href', this.datasetUrl(this.state.selected_object.dataset_pk));
        $(this.refs.experiment_link).attr('target', '_blank');
        $(this.refs.dataset_link).attr('target', '_blank');
    }

    updateMarkerVisibility() {
        var points_keys = Object.keys(this.state.points_to_trace_index);
        for (var i = 0; i < points_keys.length; i++) {

            var key = points_keys[i];
            var index = this.state.points_to_trace_index[key];

            var visibility = this.state.points_to_visibility[key];
            var update = {visible: visibility};
            Plotly.restyle('plot', update, [index]);
        }
    }

    updateVectorVisibility() {
        var update = {visible: this.state.vector_visibility};

        var indices = [];
        var keys = Object.keys(this.state.vector_to_trace_index);
        for (var i = 0; i < keys.length; i++) {
            indices.push(this.state.vector_to_trace_index[keys[i]]);
        }

        Plotly.restyle('plot', update, indices);
    }

    updateColor() {
        var points_keys = Object.keys(this.props.plot['points']);
        for (var i = 0; i < points_keys.length; i++) {

            var key = points_keys[i];
            var point_data = this.props.plot['points'][key];

            var colors = [];
            for (var j = 0; j < point_data.length; j++) {
                var point = point_data[j];
                colors.push(point['colors'][this.state.color_by]);
            }

            var index = this.state.points_to_trace_index[key];

            var update = {'marker.color': [colors]};
            Plotly.restyle('plot', update, [index]);
        }
    }

    componentDidMount(){

        var $color_by_select = $(this.refs.color_by_select);
        for (let i in this.state.color_by_choices) {
            $color_by_select.append(
                '<option val="' + i + '">' + this.state.color_by_choices[i] + '</option>');
        }

        var $hide_markers = $(this.refs.hide_markers);
        for (var i = 0; i < this.state.hide_marker_keys.length; i++) {
            var key = this.state.hide_marker_keys[i];
            var $new_div = $(
                '<input type="checkbox" id="hide_markers_' + i + '" value="' + key + '"</input> \
                <label>' + key + '&nbsp;&nbsp;</label>'
            );
            $new_div.on('click', this.update_marker_visibility.bind(this));
            $hide_markers.append($new_div);
        }

        this.drawPlotlyPCA();
        this.drawPlotlyVariance();
        this.drawPlotlyComponent(this.props.components[0], 'pc_1');
        this.drawPlotlyComponent(this.props.components[1], 'pc_2');
        this.drawPlotlyComponent(this.props.components[2], 'pc_3');
    }

    cleanDiv(div_id){
        $('#' + div_id).empty();
    }

    componentWillUnmount(){
        this.cleanDiv('plot');
        this.cleanDiv('variance_plot');
        this.cleanDiv('pc_1');
        this.cleanDiv('pc_2');
        this.cleanDiv('pc_3');
    }

    update_marker_visibility(){
        var points_to_visibility = {};

        $(this.refs.hide_markers).children('input').each(function () {
            points_to_visibility[this.value] = !this.checked;
        });

        this.setState({points_to_visibility}, this.updateMarkerVisibility);
    }

    update_vector_visibility(event){
        this.setState({vector_visibility: event.target.checked}, this.updateVectorVisibility);
    }

    change_color(event){
         this.setState({color_by: event.target.value}, this.updateColor);
    }

    render(){

        return <div>
            <div className='pca' style={{paddingTop: '20px'}}>
                <h2>Plot</h2>
                <div className='row'>
                    <div className='col-sm-9' style={{border: '1px solid black'}}>
                        <div ref='plot' id='plot'></div>
                    </div>
                    <div className='col-sm-3'>

                        <div style={{paddingTop: '10px'}}>
                            <div>Color by</div>
                            <select ref='color_by_select'
                                onChange={this.change_color.bind(this)}
                                value={this.state.color_by}>
                            </select>
                        </div>

                        {Object.keys(this.props.plot['vectors']).length > 0 &&
                            <div style={{paddingTop: '10px'}}>
                                <div>Show vectors</div>
                                <input type='checkbox' id='show_vectors' name='show_vectors' value='show_vectors'
                                    onChange={this.update_vector_visibility.bind(this)}>
                                </input>
                            </div>
                        }

                        {this.state.hide_marker_keys.length > 0 &&
                            <div style={{paddingTop: '10px'}}>
                                <div>Hide markers</div>
                                <div ref='hide_markers'></div>
                            </div>
                        }

                    </div>
                </div>
            </div>
            <div style={{paddingTop: '20px'}}>
                <h2>Components</h2>
                <ul className='nav nav-tabs' style={{paddingTop: '10px'}}>
                    <li className='active'><a data-toggle='tab' href='#variance_plot_tab'>Variance ratios</a></li>
                    <li><a data-toggle='tab' href='#pc_1_tab'>Principle component 1</a></li>
                    <li><a data-toggle='tab' href='#pc_2_tab'>Principle component 2</a></li>
                    <li><a data-toggle='tab' href='#pc_3_tab'>Principle component 3</a></li>
                </ul>
                <div className='tab-content' id='tabs'>
                    <div id='variance_plot_tab' className='tab-pane fade in active'>
                        <div id='variance_plot'></div>
                    </div>
                    <div id='pc_1_tab' className='tab-pane fade'>
                        <div id='pc_1'></div>
                    </div>
                    <div id='pc_2_tab' className='tab-pane fade'>
                        <div id='pc_2'></div>
                    </div>
                    <div id='pc_3_tab' className='tab-pane fade'>
                        <div id='pc_3'></div>
                    </div>
                </div>
            </div>
        </div>;
    }
}

PCA.defaultProps = {
    user_data: [],
};

PCA.propTypes = {
    plot: React.PropTypes.object.isRequired,
    explained_variance: React.PropTypes.array.isRequired,
    components: React.PropTypes.array.isRequired,
    user_data: React.PropTypes.array.isRequired,
};

export default PCA;
