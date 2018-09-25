function parent_folder = get_parent_folder(curr_folder)

% if 
indices=find(curr_folder=='/');
if indices(end)==numel(curr_folder)
    parent_folder=curr_folder(1:indices(end-1));
else
    parent_folder=curr_folder(1:indices(end));
end

end