import React from 'react';
import ReactDOM from 'react-dom';

import PCA from './PCA';


class PCAExplore extends React.Component {

    constructor(props) {
        super(props);

        var assembly_choices = ['--'];
        var exp_type_choices = ['--'];
        var region_choices = ['--'];

        var lookup_keys = Object.keys(this.props.pca_lookup);
        for (var i = 0; i < lookup_keys.length; i++) {
            var split = lookup_keys[i].split(':');

            var assembly = split[0];
            var exp_type = split[1];
            var region = split[2];

            if ($.inArray(assembly, assembly_choices) == -1) {
                assembly_choices.push(assembly);
            }
            if ($.inArray(exp_type, exp_type_choices) == -1) {
                exp_type_choices.push(exp_type);
            }
            if ($.inArray(region, region_choices) == -1) {
                region_choices.push(region);
            }
        }

        this.state = {
            assembly: '--',
            exp_type: '--',
            region: '--',
            pca_enabled: false,
            assembly_choices: assembly_choices,
            exp_type_choices: exp_type_choices,
            region_choices: region_choices,
        };
    }

    componentDidMount(){
        // Add assembly options
        for (var i = 0; i < this.state.assembly_choices.length; i++) {
            $(this.refs.assembly_select).append(
                '<option val="' + i + '">' + this.state.assembly_choices[i] + '</option>');
        }

        // Add experiment type options
        for (var i = 0; i < this.state.exp_type_choices.length; i++) {
            $(this.refs.experiment_type_select).append(
                '<option val="' + i + '">' + this.state.exp_type_choices[i] + '</option>');
        }

        // Add genome region options
        for (var i = 0; i < this.state.region_choices.length; i++) {
            $(this.refs.region_select).append(
                '<option val="' + i + '" style="color:grey;">' + this.state.region_choices[i] + '</option>');
        }
    }

    update_button(){
        var key = this.state.assembly + ':' +
            this.state.exp_type + ':' +
            this.state.region;
        var lookup_keys = Object.keys(this.props.pca_lookup);

        if ($.inArray(key, lookup_keys) == -1) {
            this.setState({pca_enabled: false});
        } else {
            this.setState({pca_enabled: true});
        }
    }

    update_assembly_select(){
        if (this.state.exp_type == '--') {
            var available_assemblies = this.state.assembly_choices;
        } else {
            var available_assemblies = this.props.available_assemblies[this.state.exp_type];
        }

        $(this.refs.assembly_select).find('option').each(function() {
            if ($.inArray(this.value, available_assemblies) == -1 && this.value != '--') {
                this.disabled = true;
            } else {
                this.disabled = false;
            }
        });

        this.update_button();
    }

    update_experiment_type_select(){
        if (this.state.assembly == '--') {
            var available_exp_types = this.state.exp_type_choices;
        } else {
            var available_exp_types = this.props.available_exp_types[this.state.assembly];
        }

        $(this.refs.experiment_type_select).find('option').each(function() {
            if ($.inArray(this.value, available_exp_types) == -1 && this.value != '--') {
                this.disabled = true;
            } else {
                this.disabled = false;
            }
        });

        this.update_button();
    }

    update_region_select(){
        this.update_button();
    }

    change_assembly(event){
        this.setState({
            assembly: event.target.value,
        }, this.update_experiment_type_select);
    }

    change_exp_type(event){
        this.setState({
            exp_type: event.target.value,
        }, this.update_assembly_select);
    }

    change_region(event){
        this.setState({
            region: event.target.value,
        }, this.update_region_select);
    }

    clear_pca() {
        $(this.refs.pca_container).empty();
    }

    get_pca(event){
        this.clear_pca()

        var pca_pk =
            this.props.pca_lookup[
                this.state.assembly + ':' +
                this.state.exp_type + ':' +
                this.state.region
            ];
        var pca_url = `/api/pca-plot/${pca_pk}/`;

        var cb = function(data) {
            ReactDOM.render(
                <PCA
                    plot={data.pca_plot}
                    explained_variance={data.explained_variance}
                    components={data.components}
                    user_data={this.props.user_data[pca_pk]}
                />,
                this.refs.pca_container,
            );
        };

        $.get(pca_url, cb.bind(this));
    }

    render(){
        return <div className='container-fluid' ref='explore_container'>
            <h1>PCA</h1>
            <nav className="navbar navbar-default">
                <div className="collapse navbar-collapse" id="bs-example-navbar-collapse-1">
                    <form className="navbar-search">
                        <div>
                            <p className="col-xs-1 navbar-text text-right">Assembly</p>
                            <div className="col-xs-2">
                                <select ref='assembly_select'
                                    className="form-control navbar-btn"
                                    onChange={this.change_assembly.bind(this)}
                                    value={this.state.assembly}>
                                </select>
                            </div>
                            <p className="col-xs-1 navbar-text text-right">Experiment Type</p>
                            <div className="col-xs-2">
                                <select ref='experiment_type_select'
                                    className="form-control navbar-btn"
                                    onChange={this.change_exp_type.bind(this)}
                                    value={this.state.exp_type}>
                                </select>
                            </div>
                            <p className="col-xs-1 navbar-text text-right">Genome Region</p>
                            <div className="col-xs-2">
                                <select ref='region_select'
                                    className="form-control navbar-btn"
                                    onChange={this.change_region.bind(this)}
                                    value={this.state.region}>
                                </select>
                            </div>
                            <button type="button"
                                className="btn btn-primary navbar-btn col-xs-offset-1"
                                onClick={this.get_pca.bind(this)}
                                disabled={!(this.state.pca_enabled)}>
                            Go
                            </button>
                        </div>
                    </form>
                </div>
            </nav>
            <div ref='pca_container'></div>
        </div>
    }
}

PCAExplore.propTypes = {
    pca_lookup: React.PropTypes.object.isRequired,
    available_exp_types: React.PropTypes.object.isRequired,
    available_assemblies: React.PropTypes.object.isRequired,
    available_groups: React.PropTypes.array.isRequired,
    user_data: React.PropTypes.object,
};

export default PCAExplore;
