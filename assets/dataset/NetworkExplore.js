import React from 'react';
import ReactDOM from 'react-dom';

import Network from './Network';


class NetworkExplore extends React.Component {

    constructor(props) {
        super(props);

        var organism_choices = ['--'];
        var exp_type_choices = ['--'];

        var organisms = Object.keys(this.props.organism_lookup);
        for (var i = 0; i < organisms.length; i++) {
            organism_choices.push(organisms[i])
        }

        var exp_types = Object.keys(this.props.exp_type_lookup);
        for (var i = 0; i < exp_types.length; i++) {
            exp_type_choices.push(exp_types[i])
        }

        this.state = {
            organism: '--',
            exp_type: '--',
            network_enabled: false,
            organism_choices: organism_choices,
            exp_type_choices: exp_type_choices,
        };
    }

    componentDidMount(){
        // Add organism options
        for (var i = 0; i < this.state.organism_choices.length; i++) {
            $(this.refs.organism_select).append(
                '<option val="' + i + '">' + this.state.organism_choices[i] + '</option>');
        }

        // Add experiment type options
        for (var i = 0; i < this.state.exp_type_choices.length; i++) {
            $(this.refs.exp_type_select).append(
                '<option val="' + i + '">' + this.state.exp_type_choices[i] + '</option>');
        }
    }

    clear_network(){
        $(this.refs.network_container).empty();
    }

    get_network(event){
        this.clear_network();

        var organism_pk = this.props.organism_lookup[this.state.organism];
        var exp_type_pk = this.props.exp_type_lookup[this.state.exp_type];
        var network_url = `/network/network/?organism=${organism_pk}&exp-type=${exp_type_pk}`;

        var cb = function(data) {
            ReactDOM.render(
                <Network
                    network={data}
                />,
                this.refs.network_container,
            );
        };

        $.get(network_url, cb.bind(this));
    }

    update_button(){
        if (this.state.organism == '--' || this.state.exp_type == '--') {
            this.setState({network_enabled: false});
        } else {
            var valid_organism = $.inArray(
                this.state.organism,
                this.props.available_organisms[this.state.exp_type]
            ) != -1;
            var valid_exp_type = $.inArray(
                this.state.exp_type,
                this.props.available_exp_types[this.state.organism]
            ) != -1;
            this.setState({network_enabled: (valid_organism && valid_exp_type)});
        }
    }

    update_select(){
        if (this.state.organism == '--') {
            var available_exp_types = this.state.exp_type_choices;
        } else {
            var available_exp_types = this.props.available_exp_types[this.state.organism];
        }
        $(this.refs.exp_type_select).find('option').each(function() {
            if ($.inArray(this.value, available_exp_types) == -1 && this.value != '--') {
                this.disabled = true;
            } else {
                this.disabled = false;
            }
        });

        if (this.state.exp_type == '--') {
            var available_organisms = this.state.organism_choices;
        } else {
            var available_organisms = this.props.available_organisms[this.state.exp_type];
        }
        $(this.refs.organism_select).find('option').each(function() {
            if ($.inArray(this.value, available_organisms) == -1 && this.value != '--') {
                this.disabled = true;
            } else {
                this.disabled = false;
            }
        });

        this.update_button();
    }

    change_organism(event){
        this.setState({
            organism: event.target.value,
        }, this.update_select);
    }

    change_exp_type(event){
        this.setState({
            exp_type: event.target.value,
        }, this.update_select);
    }

    render(){

        return <div className='container-fluid' ref='explore_container'>
            <h1>Network</h1>
            <nav className="navbar navbar-default">
                <div className="collapse navbar-collapse" id="bs-example-navbar-collapse-1">
                    <form className="navbar-search">
                        <div>
                            <p className="col-xs-1 navbar-text text-right">Organism</p>
                            <div className="col-xs-2">
                                <select ref='organism_select'
                                    className="form-control navbar-btn"
                                    onChange={this.change_organism.bind(this)}
                                    value={this.state.organism}>
                                </select>
                            </div>
                            <p className="col-xs-1 navbar-text text-right">Experiment Type</p>
                            <div className="col-xs-2">
                                <select ref='exp_type_select'
                                    className="form-control navbar-btn"
                                    onChange={this.change_exp_type.bind(this)}
                                    value={this.state.exp_type}>
                                </select>
                            </div>
                            <button type="button"
                                className="btn btn-primary navbar-btn col-xs-offset-1"
                                onClick={this.get_network.bind(this)}
                                disabled={!(this.state.network_enabled)}>
                            Go
                            </button>
                        </div>
                    </form>
                </div>
            </nav>
            <div ref='network_container'></div>
        </div>
    }
}

NetworkExplore.propTypes = {
    organism_lookup: React.PropTypes.object.isRequired,
    exp_type_lookup: React.PropTypes.object.isRequired,
    available_organisms: React.PropTypes.object.isRequired,
    available_exp_types: React.PropTypes.object.isRequired,
};

export default NetworkExplore;
