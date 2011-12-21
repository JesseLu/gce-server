// We assume column-major access to the 3D arrays.
#define M_IND(i,j,k,shiftx,shifty,shiftz,dx,dy,dz,xx,yy,zz) \
    int(rintf((i-shiftx)/dx)) * yy * zz + \
    int(rintf((j-shifty)/dy)) * zz + \
    int(rintf((k-shiftz)/dz))

// Store the location of our fields in constant cache.
__constant__ float *M_FIELD[{{ field_names|length }}];

// Enable the use of the names of the fields, which are kept in constant memory.
{% for fname in field_names -%}
{%- if fields[fname].isglobal -%}
#define {{ fname }} M_FIELD[{{ loop.index0 }}][0]
{%- else -%}
#define {{ fname }}(i,j,k) M_FIELD[{{ loop.index0 }}][M_IND(i,j,k 

{#- Hardcode in the appropriate shifts for each field #}
{%- for cnt in range(3) -%} 
,{{ fields[fname].global_offset[cnt] - 
    fields[fname].spacing[cnt] * fields[fname].local_offset[cnt] }} 
{%- endfor -%}

{# Hardcode in the grid spacing for each field #}
{%- for cnt in range(3) -%} 
,{{ fields[fname].spacing[cnt] }} 
{%- endfor -%}

{# Hardcode in the local dimensions of each field #}
{%- for cnt in range(3) -%} 
,{{ fields[fname].local_dims[cnt] }} 
{%- endfor -%}
)] 
{%- endif %}
{% endfor %} 

// Macro for accessing params.
#define p(i) p[int(rintf(i))]

// Shared storage for sum operations.
extern __shared__ float sum_array[]; 
