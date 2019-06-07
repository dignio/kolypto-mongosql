from __future__ import absolute_import

from sqlalchemy.orm.attributes import flag_modified

from mongosql import exc
from mongosql import MongoQuery, ModelPropertyBags
from mongosql.util import Reusable


class CrudHelper(object):
    """ Crud helper: an object that helps implement CRUD

        Create: construct sqlalchemy instances from dict
        Read: use MongoQuery for querying
        Update: update sqlalchemy instances using a dict
        Delete: use MongoQuery for deletion

        This object is supposed to be ininitialized only once;
        don't do it for every query, keep it at the class level!
    """

    # The class to use for getting structural data from a model
    _MODEL_PROPERTY_BAGS_CLS = ModelPropertyBags
    # The class to use for MongoQuery
    _MONGOQUERY_CLS = MongoQuery

    def __init__(self, model, **handler_settings):
        """ Init CRUD helper

        :param model: The model to work with
        :type model: DeclarativeMeta
        :param handler_settings: Settings for the MongoQuery used to make queries
        """
        self.model = model
        self.bags = self._MODEL_PROPERTY_BAGS_CLS.for_model(model)
        self.reusable_mongoquery = Reusable(self._MONGOQUERY_CLS(self.model, handler_settings))

    def query_model(self, query_obj=None, from_query=None):
        """ Make a MongoQuery using the provided Query Object

            Note that you have to provide the MongoQuery yourself.
            This is because it has to be properly configured with handler_settings.

            :param query_obj: The Query Object to use
            :type query_obj: dict | None
            :param from_query: An optional Query to initialize MongoQuery with
            :type from_query: Query | None
            :rtype: MongoQuery
            :raises exc.InvalidColumnError: Invalid column name specified in the Query Object by the user
            :raises exc.InvalidRelationError: Invalid relationship name specified in the Query Object by the user
            :raises exc.InvalidQueryError: There is an error in the Query Object that the user has made
            :raises exc.DisabledError: A feature is disabled; likely, due to a configuration issue. See handler_settings.
        """
        # Validate
        if not isinstance(query_obj, (dict, NoneType)):
            raise exc.InvalidQueryError('Query Object must be either an object, or null')

        # Query
        return self._query_model(query_obj or {}, from_query)  # ensure dict

    def _query_model(self, query_obj, from_query=None):
        """ Make a MongoQuery """
        return self.reusable_mongoquery.from_query(from_query).query(**query_obj)

    def _validate_columns(self, column_names, where):
        """ Validate column names

            :raises exc.InvalidColumnError: Invalid column name
        """
        unk_cols = self.bags.columns.get_invalid_names(column_names)
        if unk_cols:
            raise exc.InvalidColumnError(self.bags.model_name, unk_cols.pop(), where)

    def _validate_attributes(self, column_names, where):
        """ Validate attribute names (any)

            :raises exc.InvalidColumnError: Invalid column name
            :rtype: set[set]
        """
        column_names = set(column_names)
        unk_cols = column_names - self.bags.all_names
        if unk_cols:
            raise exc.InvalidColumnError(self.bags.model_name, unk_cols.pop(), where)
        return column_names

    def _validate_writable_attributes(self, attr_names, where):
        """ Validate attribute names (columns, properties, hybrid properties) that are writable

            This list does not include attributes like relationships and read-only properties

            :raises exc.InvalidColumnError: Invalid column name
            :rtype: set[set]
        """
        attr_names = set(attr_names)
        unk_cols = attr_names - self.bags.writable.names
        if unk_cols:
            raise exc.InvalidColumnError(self.bags.model_name, unk_cols.pop(), where)
        return attr_names


    def create_model(self, entity_dict):
        """ Create an instance from entity dict.

            This only allows to assign column properties and not relations.

            :param entity_dict: Entity dict
            :type entity_dict: dict
            :return: Created instance
            :rtype: DeclarativeMeta
            :raises InvalidQueryError: validation errors
            :raises InvalidColumnError: invalid column
        """
        # Validate
        if not isinstance(entity_dict, dict):
            raise exc.InvalidQueryError('Create model: the value has to be an object, not {}'
                                        .format(type(entity_dict)))

        # Check columns
        self._validate_columns(entity_dict.keys(), 'create')

        # Create
        return self._create_model(entity_dict)

    def _create_model(self, entity_dict):
        """ Create an instance from a dict

            This method does not validate `entity_dict`
        """
        return self.model(**entity_dict)

    def update_model(self, entity_dict, instance):
        """ Update an instance from an entity dict by merging the fields

            - Properties are copied over
            - JSON dicts are shallowly merged

            Note that because properties are *copied over*,
            this operation does not replace the entity; it merely updates the entity.

            :param entity_dict: Entity dict
            :type entity_dict: dict
            :param instance: The instance to update
            :type instance: DeclarativeMeta
            :return: New instance, updated
            :rtype: DeclarativeMeta
            :raises InvalidQueryError: validation errors
            :raises InvalidColumnError: invalid column
        """
        # Validate
        if not isinstance(entity_dict, dict):
            raise exc.InvalidQueryError('Update model: the value has to be an object, not {}'
                                        .format(type(entity_dict)))

        # Check columns
        self._validate_columns(entity_dict.keys(), 'update')

        # Update
        return self._update_model(entity_dict, instance)

    def _update_model(self, entity_dict, instance):
        """ Update an instance from an entity dict

            This method does not validate `entity_dict`
        """
        # Update
        for name, val in entity_dict.items():
            if isinstance(val, dict) and self.bags.columns.is_column_json(name):
                # JSON column with a dict: do a shallow merge
                getattr(instance, name).update(val)
                # Tell SqlAlchemy that a mutable collection was updated
                flag_modified(instance, name)
            else:
                # Other columns: just assign
                setattr(instance, name, val)

        # Finish
        return instance


class StrictCrudHelper(CrudHelper):
    """ Crud helper with limitations

        - Read-only fields can not be set: not with create, nor with update
        - Constant fields can be set initially, but never be updated
        - Defaults for Query Object provide the default values for every query, unless overridden

        The following behavior is implemented:

        * By default, all fields are writable
        * If ro_fields is provided, these fields become read-only, all other fields are writable
        * If rw_fields is provided, ony these fields are writable, all other fields are read-only
        * If const_fields, it is seen as a further limitation on rw_fields: those fields would be writable,
            but only once.

        Attributes:
            ro_fields (set[str]): The list of read-only field names
            rw_fields (set[str]): The list of writable field names
            const_fields (set[str]): The list of constant field names
            query_defaults (dict): Default values for every field of the Query Object
    """

    def __init__(self, model, ro_fields=None, rw_fields=None, const_fields=None, query_defaults=None, **handler_settings):
        """ Init a strict CRUD helper

            Note: use a **StrictCrudHelperSettingsDict() to help you with the argument names and their docs!

            :param model: The model to work with
            :param ro_fields: List of read-only property names, or a callable which gives the list
            :type ro_fields: Iterable[str] | Callable | None
            :param rw_fields: List of writable property names, or a callable which gives the list
            :type rw_fields: Iterable[str] | Callable | None
            :param const_fields: List of property names that are constant once set, or a callable which gives the list
            :type const_fields: Iterable[str] | Callable | None
            :param query_defaults: Defaults for every Query Object: Query Object will be merged into it.
            :type query_defaults: dict | None
            :param handler_settings: Settings for the MongoQuery used to make queries
        """
        super(StrictCrudHelper, self).__init__(model, **handler_settings)

        # ro, rw, const fields
        ro, rw, cn = self._init_ro_rw_cn_fields(ro_fields, rw_fields, const_fields)
        self.ro_fields = ro
        self.rw_fields = rw
        self.const_fields = cn

        # Defaults for the Query Object
        self.query_defaults = query_defaults or {}  # type: dict

        # Validate the Default Query Object
        MongoQuery(self.model).query(**self.query_defaults)

    def _init_ro_rw_cn_fields(self, ro_fields, rw_fields, cn_fields):
        """ Initialize ro_fields and rw_fields and const_fields

            :rtype: (set[str], set[str], set[str])
        """
        # Usage
        ro_provided = ro_fields is not None  # provided, even if empty
        rw_provided = rw_fields is not None
        assert not (ro_provided and rw_provided), 'Use either ro_fields or rw_fields, but not both'

        # Read-only and Read-Write fields
        ro_fields = set(call_if_callable(ro_fields)) if ro_fields is not None else set()
        rw_fields = set(call_if_callable(rw_fields)) if rw_fields is not None else set()
        cn_fields = set(call_if_callable(cn_fields)) if cn_fields is not None else set()

        # Validate
        self._validate_attributes(ro_fields, 'ro_fields')
        self._validate_writable_attributes(rw_fields, 'rw_fields')
        self._validate_writable_attributes(cn_fields, 'const_fields')

        # ro_fields
        if rw_provided:
            ro_fields = set(self.bags.all_names - rw_fields - cn_fields)

        # rw_fields
        rw_fields = self.bags.writable.names - ro_fields - cn_fields

        # Done
        return frozenset(ro_fields), frozenset(rw_fields), frozenset(cn_fields)

    def _remove_ro_fields_from(self, entity_dict):
        """ Remove read-only fields from the incoming entity dict """
        for k in set(entity_dict.keys()) & self.ro_fields:
            entity_dict.pop(k)

    def _create_model(self, entity_dict):
        # Remove ro fields
        self._remove_ro_fields_from(entity_dict)

        # Super
        return super(StrictCrudHelper, self)._create_model(entity_dict)

    def _update_model(self, entity_dict, instance):
        # Remove ro fields
        self._remove_ro_fields_from(entity_dict)

        # Super
        return super(StrictCrudHelper, self)._update_model(entity_dict, instance)

    def _query_model(self, query_obj, from_query=None):
        # Default Query Object
        if self.query_defaults:
            query_obj = shallow_merge_dicts(self.query_defaults, query_obj or {})

        # Super
        return super(StrictCrudHelper, self)._query_model(query_obj, from_query=from_query)


NoneType = type(None)


def call_if_callable(v):
    """ Preprocess a value: return it ; but call it, if it's a lambda (for late binding) """
    return v() if callable(v) else v


def shallow_merge_dicts(d1, d2):
    """ Merge two dicts, d2 into d1, shallowly """
    d = {}
    d.update(d1)
    d.update(d2)
    return d
