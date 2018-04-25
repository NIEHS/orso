import React from 'react';
import ReactDOM from 'react-dom';

import Network from './Network';


class NetworkExplore extends React.Component {

    clear_network(){
        $(this.refs.network_container).empty();
    }

    get_network(event){
        this.clear_network();

        var network_pk = $(this.refs.network_select).val();
        var network_url = `/network/api/network/${network_pk}/`;

        var cb = function(data) {
            ReactDOM.render(
                <Network
                    network={data.network}
                />,
                this.refs.network_container,
            );
        };

        $.get(network_url, cb.bind(this));
    }

    render(){
        var lookup_keys = Object.keys(this.props.network_lookup);
        var options = [];
        for (var i = 0; i < lookup_keys.length; i++) {
            var key = lookup_keys[i];
            var pk = this.props.network_lookup[key];
            options.push(<option key={pk} value={pk}>{key}</option>);
        }

        return <div className='container-fluid' ref='explore_container'>
            <h1>Network</h1>
            <nav className="navbar navbar-default">
                <div className="collapse navbar-collapse" id="bs-example-navbar-collapse-1">
                    <form className="navbar-search">
                        <div>
                            <p className="col-xs-1 navbar-text text-right">Organism</p>
                            <div className="col-xs-2">
                                <select ref='network_select' className="form-control navbar-btn">
                                    {options}
                                </select>
                            </div>
                            <button type="button"
                                className="btn btn-primary navbar-btn col-xs-offset-1"
                                onClick={this.get_network.bind(this)}>
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
    network_lookup: React.PropTypes.object.isRequired,
};

export default NetworkExplore;
